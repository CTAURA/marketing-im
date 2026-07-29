#!/usr/bin/env python3
"""Stream the official Yelp dataset archive into compact intermediates.

The Yelp open dataset ships as ``Yelp-JSON.zip`` (~4.35 GB) containing a single
member ``Yelp JSON/yelp_dataset.tar`` which -- despite the name -- is gzip
compressed.  Fully extracting it costs ~10 GB of disk on top of the zip.  This
machine does not have that much headroom, so we never materialise the JSON
files: we stream the tar and emit small, purpose-built intermediates instead.

Member order inside the tar (verified) is::

    Dataset_User_Agreement.pdf
    yelp_academic_dataset_business.json     ~118 MB
    yelp_academic_dataset_checkin.json      ~287 MB
    yelp_academic_dataset_review.json       ~5.3 GB
    yelp_academic_dataset_tip.json
    yelp_academic_dataset_user.json         ~3.3 GB

That ordering is what makes a *single* pass possible: by the time ``user.json``
arrives we already know which users produced reviews, so each user's ``friends``
list can be filtered down to reviewing users on the fly instead of being buffered.

Outputs (into ``--out-dir``):

``businesses.tsv``   ``business_id <TAB> cat_id,cat_id,...``
``reviews.tsv``      ``user_id <TAB> business_id <TAB> epoch_seconds``
``friends.tsv``      ``user_id <TAB> friend_id,friend_id,...``  (reviewers only)
``categories.tsv``   ``cat_id <TAB> category name``
``stream_meta.json`` counts + provenance

Section 8 of SPEC.md: "we assume the time at which the user reviews the item is
the time when the item is purchased" and "we disregard the unfrequent behaviors
of repeated review for the same item" -- the de-duplication of (user, item) pairs
happens here, keeping the earliest review.

Standard library only.  Python 3.9 compatible.
"""

import argparse
import calendar
import io
import json
import os
import sys
import tarfile
import time
import zipfile

BUSINESS = "yelp_academic_dataset_business.json"
REVIEW = "yelp_academic_dataset_review.json"
USER = "yelp_academic_dataset_user.json"

# Members we care about, in the order they appear in the archive.  Once the last
# one has been seen we can stop reading the stream entirely.
WANTED = (BUSINESS, REVIEW, USER)


def parse_yelp_date(s):
    """'2016-03-09 20:56:34' -> epoch seconds (UTC).

    Yelp timestamps carry no timezone; they are treated as UTC consistently,
    which is harmless because only *differences* matter (Definition 1's Delta).
    """
    # Much faster than datetime.strptime on millions of rows.
    try:
        y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
        h = int(s[11:13]); mi = int(s[14:16]); se = int(s[17:19])
    except (ValueError, IndexError):
        return None
    try:
        return calendar.timegm((y, mo, d, h, mi, se, 0, 1, 0))
    except (ValueError, OverflowError):
        return None


def open_tar_stream(zip_path):
    """Yield a streaming tarfile over the gzip'd tar inside the zip.

    ``zipfile`` gives us a file-like object for the tar member; ``tarfile`` in
    ``r|gz`` (stream) mode consumes it sequentially without seeking, so nothing
    is ever written to disk.
    """
    zf = zipfile.ZipFile(zip_path)
    inner = None
    for name in zf.namelist():
        if name.endswith(".tar"):
            inner = name
            break
    if inner is None:
        raise SystemExit("no .tar member found inside %s" % zip_path)
    raw = zf.open(inner, "r")
    # Buffer generously: the underlying zip member is DEFLATE'd, and tarfile
    # issues many small reads.
    buffered = io.BufferedReader(raw, buffer_size=1 << 22)
    return zf, tarfile.open(fileobj=buffered, mode="r|gz")


def stream(zip_path, out_dir, progress_every=1000000):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    cat_ids = {}          # category string -> int id
    reviewers = set()     # user_ids that produced at least one review
    seen_pairs = set()    # (user_id, business_id) already emitted
    counts = {"businesses": 0, "reviews": 0, "reviews_kept": 0,
              "users": 0, "users_kept": 0, "friend_links": 0}

    f_bus = open(os.path.join(out_dir, "businesses.tsv"), "w")
    f_rev = open(os.path.join(out_dir, "reviews.tsv"), "w")
    f_fri = open(os.path.join(out_dir, "friends.tsv"), "w")

    zf, tar = open_tar_stream(zip_path)
    remaining = set(WANTED)
    try:
        for member in tar:
            base = os.path.basename(member.name)
            if base not in remaining:
                continue
            remaining.discard(base)
            fh = tar.extractfile(member)
            if fh is None:
                continue
            sys.stderr.write("[stream] %s (%.1f MB) at t=%.0fs\n"
                             % (base, member.size / 1e6, time.time() - t0))
            sys.stderr.flush()

            if base == BUSINESS:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    bid = rec.get("business_id")
                    if not bid:
                        continue
                    cats = rec.get("categories") or ""
                    ids = []
                    for c in cats.split(","):
                        c = c.strip()
                        if not c:
                            continue
                        cid = cat_ids.get(c)
                        if cid is None:
                            cid = len(cat_ids)
                            cat_ids[c] = cid
                        ids.append(cid)
                    counts["businesses"] += 1
                    f_bus.write("%s\t%s\n" % (bid, ",".join(str(x) for x in sorted(set(ids)))))

            elif base == REVIEW:
                n = 0
                for line in fh:
                    n += 1
                    if progress_every and n % progress_every == 0:
                        sys.stderr.write("  reviews %dM  t=%.0fs\n"
                                         % (n // 1000000, time.time() - t0))
                        sys.stderr.flush()
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    uid = rec.get("user_id"); bid = rec.get("business_id")
                    if not uid or not bid:
                        continue
                    ts = parse_yelp_date(rec.get("date") or "")
                    if ts is None:
                        continue
                    key = (uid, bid)
                    # SPEC.md section 8: drop repeated reviews of the same item.
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    reviewers.add(uid)
                    counts["reviews_kept"] += 1
                    f_rev.write("%s\t%s\t%d\n" % (uid, bid, ts))
                counts["reviews"] = n
                # The pair set is only needed for de-duplication; release it
                # before user.json (the biggest member) starts streaming.
                seen_pairs = set()

            elif base == USER:
                n = 0
                for line in fh:
                    n += 1
                    if progress_every and n % progress_every == 0:
                        sys.stderr.write("  users %dM  t=%.0fs\n"
                                         % (n // 1000000, time.time() - t0))
                        sys.stderr.flush()
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    uid = rec.get("user_id")
                    if not uid or uid not in reviewers:
                        continue
                    fr = rec.get("friends") or ""
                    if fr == "None":
                        fr = ""
                    kept = []
                    for x in fr.split(","):
                        x = x.strip()
                        # Restrict to users who actually reviewed something --
                        # a friendship link to a user with no adoption log can
                        # never participate in a potential-influence log.
                        if x and x in reviewers:
                            kept.append(x)
                    if not kept:
                        continue
                    counts["users_kept"] += 1
                    counts["friend_links"] += len(kept)
                    f_fri.write("%s\t%s\n" % (uid, ",".join(kept)))
                counts["users"] = n

            if not remaining:
                break
    finally:
        try:
            tar.close()
        except Exception:
            pass
        zf.close()
        f_bus.close(); f_rev.close(); f_fri.close()

    with open(os.path.join(out_dir, "categories.tsv"), "w") as fh:
        for c, cid in sorted(cat_ids.items(), key=lambda kv: kv[1]):
            fh.write("%d\t%s\n" % (cid, c))

    meta = {
        "source": "official Yelp open dataset (Yelp-JSON.zip), streamed",
        "note": ("NOT the retired Yelp Dataset Challenge 2014 snapshot used by the "
                 "paper; that release is no longer distributed. Absolute numbers "
                 "therefore differ from the paper's Yelp figures."),
        "n_categories": len(cat_ids),
        "n_reviewers": len(reviewers),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta.update(counts)
    with open(os.path.join(out_dir, "stream_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    sys.stderr.write("[stream] done in %.0fs\n%s\n"
                     % (time.time() - t0, json.dumps(meta, indent=2)))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="data/raw/yelp/Yelp-JSON.zip")
    ap.add_argument("--out-dir", default="data/raw/yelp/stream")
    ap.add_argument("--progress-every", type=int, default=1000000)
    args = ap.parse_args(argv)
    if not os.path.exists(args.zip):
        raise SystemExit("missing %s -- download it from "
                         "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
                         % args.zip)
    stream(args.zip, args.out_dir, args.progress_every)


if __name__ == "__main__":
    main()
