import os
import cv2
import fitz
import numpy
from PIL import Image, PngImagePlugin
from subprocess import Popen, PIPE, DEVNULL

ITERATIONS = 1

PROFILE = ["files/profile/" + file for file in os.listdir("files/profile")]
DRILLS = ["files/drills/" + file for file in os.listdir("files/drills")]
TOP = ["files/top/" + file for file in os.listdir("files/top")]
PADS = ["files/pads/" + file for file in os.listdir("files/pads")]

print("# Generate top layers pdf")
p = Popen(
    f".\\gerbv\\gerbv.exe -b#ffffff {' -f#000000 '*(len(PROFILE)+len(TOP)+len(PADS))} {' '.join(PROFILE)} {' '.join(TOP)} {' '.join(PADS)} -xpdf -o{os.path.abspath('out/top.pdf')}",
    stdin=PIPE,
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print("#DONE")

print("# Generate drills layer pdf")
Popen(
    f".\\gerbv\\gerbv.exe -b#ffffff {' -f#000000 '*(len(PROFILE)+len(DRILLS))} {' '.join(PROFILE)} {' '.join(DRILLS)} -xpdf -o{os.path.abspath('out/drills.pdf')}",
    stdin=PIPE,
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print("#DONE")

print("# Generate profile layer pdf")
Popen(
    f".\\gerbv\\gerbv.exe -b#ffffff {' -f#000000 '*(len(PROFILE))} {' '.join(PROFILE)} -xpdf -o{os.path.abspath('out/profile.pdf')}",
    stdin=PIPE,
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print("#DONE")

print("# Generate pads pdf")
Popen(
    f".\\gerbv\\gerbv.exe -b#ffffff {' -f#000000 '*(len(PROFILE) + len(PADS))} {' '.join(PROFILE)} {' '.join(PADS)} -xpdf -o{os.path.abspath('out/pads.pdf')}",
    stdin=PIPE,
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print("#DONE")

p.communicate(input=b"\n")

img_top = None
img_pads = None
img_drills = None
img_profile = None

print("#Converting pdf files to image")
page_top = fitz.open("out/top.pdf")
page_top = page_top.load_page(0).get_pixmap(dpi=1000)
page_pads = fitz.open("out/pads.pdf")
page_pads = page_pads.load_page(0).get_pixmap(dpi=1000)
page_drills = fitz.open("out/drills.pdf")
page_drills = page_drills.load_page(0).get_pixmap(dpi=1000)
page_profile = fitz.open("out/profile.pdf")
page_profile = page_profile.load_page(0).get_pixmap(dpi=1000)

print(page_top)

img_top = cv2.cvtColor(
    numpy.array(
        Image.frombytes(
            "RGB", [page_top.width, page_top.height], page_top.samples
        ).convert("RGB")
    ),
    cv2.COLOR_RGB2BGR,
)
img_pads = cv2.cvtColor(
    numpy.array(
        Image.frombytes(
            "RGB", [page_pads.width, page_pads.height], page_pads.samples
        ).convert("RGB")
    ),
    cv2.COLOR_RGB2BGR,
)
img_drills = cv2.cvtColor(
    numpy.array(
        Image.frombytes(
            "RGB", [page_drills.width, page_drills.height], page_drills.samples
        ).convert("RGB")
    ),
    cv2.COLOR_RGB2BGR,
)
img_profile = cv2.cvtColor(
    numpy.array(
        Image.frombytes(
            "RGB", [page_profile.width, page_profile.height], page_profile.samples
        ).convert("RGB")
    ),
    cv2.COLOR_RGB2BGR,
)

print("#DONE")

print("#image processing n stuff")
img_top = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
img_pads = cv2.cvtColor(img_pads, cv2.COLOR_BGR2GRAY)
img_drills = cv2.cvtColor(img_drills, cv2.COLOR_BGR2GRAY)
img_profile = cv2.cvtColor(img_profile, cv2.COLOR_BGR2GRAY)

_, img_top = cv2.threshold(img_top, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
_, img_pads = cv2.threshold(img_pads, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
_, img_drills = cv2.threshold(
    img_drills, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
)
_, img_profile = cv2.threshold(
    img_profile, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
)

contours_profile, _ = cv2.findContours(
    img_profile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

mask = numpy.zeros(img_profile.shape, img_profile.dtype)
mask = cv2.drawContours(mask, contours_profile, -1, (255, 255, 255), -1)

mask_eroded = cv2.erode(mask.copy(), numpy.ones((5, 5), numpy.uint8), iterations=ITERATIONS)  # type: ignore

img_top = cv2.bitwise_and(img_top, img_top, mask=mask_eroded)
img_pads = cv2.bitwise_and(img_pads, img_pads, mask=mask_eroded)
img_pads = cv2.bitwise_not(img_pads)
img_drills = cv2.bitwise_and(img_drills, img_drills, mask=mask_eroded)
img_drills = cv2.bitwise_not(img_drills)

Image.fromarray(cv2.cvtColor(img_top, cv2.COLOR_BGR2RGB)).save(
    "top.png",
    dpi=(1000, 1000),
)
Image.fromarray(cv2.cvtColor(img_drills, cv2.COLOR_BGR2RGB)).save(
    "drills.png",
    dpi=(1000, 1000),
)
Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)).save(
    "profile.png",
    dpi=(1000, 1000),
)
Image.fromarray(cv2.cvtColor(img_pads, cv2.COLOR_BGR2RGB)).save(
    "solder_mask.png",
    dpi=(1000, 1000),
)

print("#finished")
