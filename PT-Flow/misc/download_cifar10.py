"""Fetch CIFAR-10 from Google Drive into CIFAR10_PATH.

    python -m misc.download_cifar10

Downloads `cifar-10-python.tar.gz`, verifies its MD5, and extracts it so that
`CIFAR10_PATH/cifar-10-batches-py/` exists -- which is the layout
`torchvision.datasets.CIFAR10(root=CIFAR10_PATH)` expects.

Done once, up front, on rank 0 only: `_build_dataset` constructs CIFAR10 with
`download=False` because a download racing across DistributedSampler workers
corrupts the archive.
"""

import hashlib
import os
import sys
import tarfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.env import CIFAR10_PATH

GDRIVE_ID = "1AMESa9etn7VmnXb3GPbO2DWYttULbRvY"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
ARCHIVE_NAME = "cifar-10-python.tar.gz"


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if not CIFAR10_PATH:
        raise SystemExit("Set CIFAR10_PATH in utils/env.py first.")
    os.makedirs(CIFAR10_PATH, exist_ok=True)
    batches = os.path.join(CIFAR10_PATH, "cifar-10-batches-py")
    archive = os.path.join(CIFAR10_PATH, ARCHIVE_NAME)

    if os.path.exists(os.path.join(batches, "data_batch_1")):
        print(f"CIFAR-10 already extracted at {batches}")
        return

    if not os.path.exists(archive) or md5_of(archive) != ARCHIVE_MD5:
        try:
            import gdown
        except ImportError:
            raise SystemExit("pip install gdown")
        print(f"downloading {ARCHIVE_NAME} from Google Drive ...")
        gdown.download(id=GDRIVE_ID, output=archive, quiet=False)
        if not os.path.exists(archive):
            raise SystemExit(
                "Drive download failed. Check the pod has outbound internet, "
                "or fetch the file manually to " + archive
            )

    got = md5_of(archive)
    if got != ARCHIVE_MD5:
        raise SystemExit(
            f"MD5 mismatch: got {got}, expected {ARCHIVE_MD5}.\n"
            "Drive usually serves an HTML interstitial instead of the file when "
            "the link is not shared publicly -- delete the file and retry."
        )
    print(f"md5 ok ({got})")

    print(f"extracting into {CIFAR10_PATH} ...")
    with tarfile.open(archive, "r:gz") as tar:
        # Refuse absolute or parent-relative members rather than trusting the
        # archive to stay inside the target directory.
        for m in tar.getmembers():
            target = os.path.realpath(os.path.join(CIFAR10_PATH, m.name))
            if not target.startswith(os.path.realpath(CIFAR10_PATH) + os.sep):
                raise SystemExit(f"refusing unsafe archive member: {m.name}")
        tar.extractall(CIFAR10_PATH)

    if not os.path.exists(os.path.join(batches, "data_batch_1")):
        raise SystemExit(f"extraction did not produce {batches}/data_batch_1")

    n = sum(os.path.getsize(os.path.join(batches, f)) for f in os.listdir(batches))
    print(f"CIFAR-10 ready at {batches} ({n/1e6:.0f} MB)")
    print("50,000 train / 10,000 test, 10 classes")


if __name__ == "__main__":
    main()
