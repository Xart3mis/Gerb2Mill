import os
import cv2
import numpy
from io import BytesIO
from PIL import Image
from subprocess import Popen, PIPE, DEVNULL
from pdf2image.pdf2image import convert_from_path

PROFILE = ["files/profile/" + file for file in os.listdir("files/profile")]
DRILLS = ["files/drills/" + file for file in os.listdir("files/drills")]
TOP = ["files/top/" + file for file in os.listdir("files/top")]
PADS = ["files/pads/" + file for file in os.listdir("files/pads")]

print("# Generate top layers pdf")
p = Popen(
    f".\\gerbv\\gerbv.exe -b#ffffff -D1000 {' -f#000000 '*(len(PROFILE)+len(TOP)+len(PADS))} {' '.join(PROFILE)} {' '.join(TOP)} {' '.join(PADS)} -xpdf -o{os.path.abspath('out/top.pdf')}",
    stdin=PIPE,
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print("#DONE")

print("# Generate drills layer pdf")
Popen(
    f".\\gerbv\\gerbv.exe -b#ffffff -D1000 {' -f#000000 '*(len(PROFILE)+len(DRILLS))} {' '.join(PROFILE)} {' '.join(DRILLS)} -xpdf -o{os.path.abspath('out/drills.pdf')}",
    stdin=PIPE,
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print("#DONE")

print("# Generate profile layer pdf")
Popen(
    f".\\gerbv\\gerbv.exe -b#ffffff -D1000 {' -f#000000 '*(len(PROFILE))} {' '.join(PROFILE)} -xpdf -o{os.path.abspath('out/profile.pdf')}",
    stdin=PIPE,
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)

print("# Generate pads pdf")
Popen(
    f".\\gerbv\\gerbv.exe -b#ffffff -D1000 {' -f#000000 '*(len(PROFILE) + len(PADS))} {' '.join(PROFILE)} {' '.join(PADS)} -xpdf -o{os.path.abspath('out/pads.pdf')}",
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
page_top = convert_from_path("out/top.pdf", poppler_path="poppler/")
page_pads = convert_from_path("out/pads.pdf", poppler_path="poppler/")
page_drills = convert_from_path("out/drills.pdf", poppler_path="poppler/")
page_profile = convert_from_path("out/profile.pdf", poppler_path="poppler/")


with BytesIO() as f:
    page_top[0].save(f, format="jpeg")
    f.seek(0)
    img_top = cv2.cvtColor(numpy.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
with BytesIO() as f:
    page_drills[0].save(f, format="jpeg")
    f.seek(0)
    img_drills = cv2.cvtColor(
        numpy.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR
    )
with BytesIO() as f:
    page_profile[0].save(f, format="jpeg")
    f.seek(0)
    img_profile = cv2.cvtColor(
        numpy.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR
    )
with BytesIO() as f:
    page_pads[0].save(f, format="jpeg")
    f.seek(0)
    img_pads = cv2.cvtColor(
        numpy.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR
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

mask_eroded = cv2.erode(mask.copy(), numpy.ones((10, 10), numpy.uint8), iterations=2)  # type: ignore

img_top = cv2.bitwise_and(img_top, img_top, mask=mask_eroded)
img_pads = cv2.bitwise_and(img_pads, img_pads, mask=mask_eroded)
img_pads = cv2.bitwise_not(img_pads)
img_drills = cv2.bitwise_and(img_drills, img_drills, mask=mask_eroded)
img_drills = cv2.bitwise_not(img_drills)

cv2.imwrite("top.png", img_top)
cv2.imwrite("drills.png", img_drills)
cv2.imwrite("profile.png", mask)
cv2.imwrite("solder_mask.png", img_pads)
print("#finished")
