#!/usr/bin/python3
# 2017.11.26 23:27:12 CST

## Find object by orb features matching

import numpy as np
import cv2
imgname = "box.png"          # query image (small object)
imgname2 = "box_in_scene.png" # train image (large scene)

MIN_MATCH_COUNT = 4

## Create ORB object and BF object(using HAMMING)
orb = cv2.ORB_create()
img1 = cv2.imread(imgname)
img2 = cv2.imread(imgname2)

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

## Find the keypoints and descriptors with ORB
kpts1, descs1 = orb.detectAndCompute(gray1,None)
kpts2, descs2 = orb.detectAndCompute(gray2,None)

## match descriptors and sort them in the order of their distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descs1, descs2)
dmatches = sorted(matches, key = lambda x:x.distance)

## extract the matched keypoints
src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

## find homography matrix and do perspective transform
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

## draw found regions
img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
cv2.imshow("found", img2)

## draw match lines
res = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20],None,flags=2)

cv2.imshow("orb_match", res);

cv2.waitKey();cv2.destroyAllWindows()
