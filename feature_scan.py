#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

def to_cartesian(r, phi):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x, y

def wraptopi(phi):
    return np.mod(phi + np.pi, 2*np.pi) - np.pi

def wraptohalfpi(phi):
    return np.mod(phi + np.pi/2, np.pi) - np.pi/2

def dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# PARAMETERS

rmin = 0.1  # [m] - minimum range distance
rmax = 1.5  # [m] - maximum range distance
Lmin = 0.15 # [m] - minimum line length
Pmin = 10   # [-] - minimum number of scan points in line
Snum = 6    # [-] - number of scan points in seed
eps  = 0.02 # [m] - maximum distance from point in seed segment to fitted line
delt = 0.04 # [m] - maximum distance from point to point in seed segment
thres = 0.15 # [m] - maximum distance between two end points to be counted as a corner point
phimin = np.pi/6 # [rad] - minimum angular difference between two consecutive lines for their intersection to be counted as a corned point
corspd = 0.1 # [m] - maximum distance between end point differences between two lines for correspondence

# SCAN OBJECTS

class ScanFeature(object):

    def __init__(self, r, phi=None, frame="local"):
        """
        r - 1D array of range readings

        phi - [op.] 1D array of angle readings (default: -120 to 120 degrees)
        """

        if phi == None:
            phi = np.linspace(-2*np.pi/3, 2*np.pi/3, len(r))

        mask = (r < rmax) & (r > rmin)
        self.r = r[mask]
        self.phi = phi[mask]
        self.n = len(self.r) # number of scan points
        self.lines = []
        self.kpoints = []
        self.frame = frame

        self.find_lines()
        for line in self.lines:
            line.calc_params()
        self.clean_lines()
        self.find_keypoints()

        if frame == "global":
            self.xt = 0
            self.yt = 0
            for line in self.lines:
                line.frame = "global"
                line.it = 1
            for kpoint in self.kpoints:
                kpoint.frame = "global"

    def detect_seed(self, i):
        j = i + Snum
        seed = Seed(i, j, self.r[i:j], self.phi[i:j])
        for k in range(i, j):
            d1 = seed.p2p_dist(self.r[k], self.phi[k])
            if d1 > delt:
                return None
            d2 = seed.p2l_dist(self.r[k], self.phi[k])
            if d2 > eps:
                return None

        dx = np.diff(seed.x)
        dy = np.diff(seed.y)
        dists = np.sqrt(dx**2 + dy**2)
        if np.any(dists > delt):
            return None

        return seed

    def grow_region(self, seed):
        line = seed.to_line()

        while line.end+1 < self.n and line.p2l_dist(self.r[line.end+1], self.phi[line.end+1]) < eps:
            line = Line(line.start, line.end+1, self.r[line.start:line.end+1], self.phi[line.start:line.end+1])

        while line.start-1 >= 0 and line.p2l_dist(self.r[line.start-1], self.phi[line.start-1]) < eps:
            line = Line(line.start-1, line.end, self.r[line.start-1:line.end], self.phi[line.start-1:line.end])

        # Calculate line parameters
        line.calc_params()

        if line.length >= Lmin and line.P >= Pmin:
            return line
        else:
            return None

    def overlap_processing(self, lines):
        if len(lines) == 1:
            return lines
        elif lines[1].start > lines[0].end:
            return [lines[0]] + self.overlap_processing(lines[1:])
        elif lines[1].start == lines[0].end:
            lines[1] = Line(lines[1].start-1, lines[1].end, self.r[lines[1].start-1:lines[1].end], self.phi[lines[1].start-1:lines[1].end])
            return self.overlap_processing(lines)
        else:
            for k in range(lines[1].start, lines[0].end):
                dist_i_k = lines[0].p2l_dist(self.r[k], self.phi[k])
                dist_j_k = lines[1].p2l_dist(self.r[k], self.phi[k])
                if dist_i_k >= dist_j_k:
                    break

            if (k-1) - lines[0].start >= Pmin:
                lines[0] = Line(lines[0].start, k-1, self.r[lines[0].start:k-1], self.phi[lines[0].start:k-1])
                if lines[1].end - k >= Pmin:
                    lines[1] = Line(k, lines[1].end, self.r[k:lines[1].end], self.phi[k:lines[1].end])
                    return self.overlap_processing(lines)
                else:
                    del lines[1]
                    return self.overlap_processing(lines)
            else:
                lines[1] = Line(k, lines[1].end, self.r[k:lines[1].end], self.phi[k:lines[1].end])
                del lines[0]
                return self.overlap_processing(lines)

    def find_lines(self):
        i = 0
        while i <= self.n - Pmin:
            seed = self.detect_seed(i)
            if seed == None:
                i = i + 1
            else:

                line = self.grow_region(seed)
                if line == None:
                    i = i + 1
                else:
                    self.lines.append(line)
                    i = line.end

        overlap = True
        while overlap:
            start_idx = []
            end_idx = []
            for line in self.lines:
                start_idx.append(line.start)
                end_idx.append(line.end)
            start = np.array(start_idx[1:])
            end = np.array(end_idx[:-1])
            if np.any(start - end <= 0):
                self.lines = self.overlap_processing(self.lines)
            else:
                overlap = False

    def clean_lines(self):
        merged_lines = []
        if len(self.lines) == 1:
            merged_lines.append(self.lines[0])
        elif len(self.lines) > 1:
            skip = False
            for i in range(len(self.lines)-1):
                if not skip:
                    a_diff = np.abs(self.lines[i].a - self.lines[i+1].a)
                    b_diff = np.abs(self.lines[i].b - self.lines[i+1].b)
                    c_diff = np.abs(self.lines[i].c - self.lines[i+1].c)

                    if np.all(np.array([a_diff, b_diff, c_diff]) < 0.2) \
                    and dist(self.lines[i].x_end, self.lines[i].y_end, self.lines[i+1].x_start, self.lines[i+1].y_start) < delt:
                        skip = True
                        merged_lines.append(Line(self.lines[i].start,
                                                 self.lines[i+1].end,
                                                 self.r[self.lines[i].start:self.lines[i+1].end],
                                                 self.phi[self.lines[i].start:self.lines[i+1].end],
                                                 calc=True))
                    else:
                        merged_lines.append(self.lines[i])
                else:
                    skip = False

            if not skip:
                merged_lines.append(self.lines[-1])

        cleaned_lines = []
        for line in merged_lines:
            if abs(line.c) > 0.1:
                cleaned_lines.append(line)

        valid_lines = []
        for line in cleaned_lines:
            if line.length >= Lmin and line.P >= Pmin:
                valid_lines.append(line)

        self.lines = valid_lines

    def find_keypoints(self):

        # Initiate candidate keypoints
        try:
            cands = np.vstack([np.array([[line.x_start, line.y_start], [line.x_end, line.y_end]]) for line in self.lines])
        except: # if there is only one line
            return None
        cand_idx = sum([[line.start, line.end] for line in self.lines], [])

        # Find corner points
        corner_idx = []
        for i in range(len(self.lines) - 1):
            end_dist = np.sqrt((self.lines[i].x_end - self.lines[i+1].x_start)**2 +
                               (self.lines[i].y_end - self.lines[i+1].y_start)**2)
            if end_dist < thres and np.abs(self.lines[i].heading - self.lines[i+1].heading) > phimin:
                corner_idx.extend([self.lines[i].end, self.lines[i+1].start])
                x_corner = (self.lines[i].b**2*self.lines[i+1].x_start -
                            self.lines[i].a*self.lines[i].b*self.lines[i+1].y_start -
                            self.lines[i].a*self.lines[i].c) / (self.lines[i].a**2 + self.lines[i].b**2)
                y_corner = (self.lines[i].a**2*self.lines[i+1].y_start -
                            self.lines[i].a*self.lines[i].b*self.lines[i+1].x_start -
                            self.lines[i].b*self.lines[i].c) / (self.lines[i].a**2 + self.lines[i].b**2)
                self.kpoints.append(Keypoint(x_corner, y_corner, point_type="corner"))

        # Disregard occluded candidate points
        occluded_idx = []
        for idx in cand_idx:
            for inc in range(1,5):
                try:
                    if self.r[idx] - self.r[idx-inc] > 0.2 or self.r[idx] - self.r[idx+inc] > 0.2:
                        occluded_idx.append(idx)
                        break
                except:
                    break

        # Disregard candidate points at the edges of scan range
        blindspot_idx = [cand_idx[0], cand_idx[-1]]

        # Find true end keypoints
        not_kpoints_idx = corner_idx + occluded_idx + blindspot_idx
        kpoints_idx = np.array(list(set(cand_idx) - set(not_kpoints_idx)))

        for idx in kpoints_idx:
            line = self.lines[cand_idx.index(idx) // 2]
            if cand_idx.index(idx) % 2 == 0: # if idx is even, then start point is a keypoint
                self.kpoints.append(Keypoint(line.x_start, line.y_start, point_type="end"))
            else:                            # otherwise, end point is a keypoint
                self.kpoints.append(Keypoint(line.x_end, line.y_end, point_type="end"))

    def to_global(self, phit, xt, yt):
        """
        Given state {xt, yt, phit} of the vehicle, transform the scan to the global frame.
        """
        if self.frame == "global":
            print("Warning: transforming scan from global frame!")

        self.phi += phit
        self.xt = xt
        self.yt = yt

        for line in self.lines:
            line.to_global(phit, xt, yt)

        for point in self.kpoints:
            point.to_global(phit, xt, yt)

        self.frame = "global"

    def plot_all_lines(self, pc=False, keypoints=False):
        if self.frame == "global":
            xp, yp = to_cartesian(self.r, self.phi)
            xp += self.xt
            yp += self.yt
        if pc:
            xp, yp = to_cartesian(self.r, self.phi)
            plt.plot(xp, yp, '.', markersize=0.5, label="Point cloud")

        for i, line in enumerate(self.lines):
            plt.plot([line.x_start, line.x_end], [line.y_start, line.y_end], linewidth=2, label="Line segment {0}".format(i+1))

        if keypoints and self.kpoints != []:
            plot_end = False
            plot_corner = False
            for p in self.kpoints:
                if p.type == "end":
                    if not plot_end:
                        plt.plot(p.x, p.y, 'go', label="End keypoint")
                        plot_end = True
                    else:
                        plt.plot(p.x, p.y, 'go')
                elif p.type == "corner":
                    if not plot_corner:
                        plt.plot(p.x, p.y, 'gs', label="Corner keypoint")
                        plot_corner = True
                    else:
                        plt.plot(p.x, p.y, 'gs')

        plt.axis("equal")
        plt.legend(loc="upper left",fontsize=6)
        plt.show()

# POINT OBJECTS

class Point(object):

    def __init__(self, x, y, frame="local"):
        self.x = x
        self.y = y
        self.frame = frame

    def to_global(self, phit, xt, yt):
        """
        Given state {xt, yt, phit} of the vehicle, transform the point cloud to the global frame.
        """
        if self.frame == "global":
            print("Warning: transforming point from global frame!")

        r = np.sqrt(self.x**2 + self.y**2)
        phi = np.arctan2(self.y, self.x)
        self.x = r*np.cos(phi + phit) + xt
        self.y = r*np.sin(phi + phit) + yt
        self.frame = "global"

class Keypoint(Point):

    def __init__(self, x, y, point_type="end", frame="local"):
        self.x = x
        self.y = y
        self.type = point_type # either "end" or "corner"
        self.frame = frame

# LINE OBJECTS

class Line(object):

    obj_type = "Line"

    def __init__(self, i, j, r, phi, calc=False, frame="local"):
        self.start = i
        self.end = j
        self.r = r
        self.phi = phi
        self.frame = frame
        self.x, self.y = to_cartesian(self.r, self.phi)

        self.fit_line(self.x, self.y)
        if calc:
            self.calc_params()

    def fit_line(self, x, y):
        if np.abs(x[-1] - x[0]) >= np.abs(y[-1] - y[0]):
            self.a, self.c = np.polyfit(x, y, 1)
            self.b = -1
        else:
            self.b, self.c = np.polyfit(y, x, 1)
            self.a = -1
        self.heading = np.arctan(-self.a/self.b)

    def calc_params(self):
        self.P = self.end - self.start + 1 # number of points

        # CHANGE THESE TO POINT OBJECTS
        self.x_start = (self.b**2*self.x[0] - self.a*self.b*self.y[0] - self.a*self.c)/(self.a**2 + self.b**2)
        self.y_start = (self.a**2*self.y[0] - self.a*self.b*self.x[0] - self.b*self.c)/(self.a**2 + self.b**2)
        self.x_end = (self.b**2*self.x[-1] - self.a*self.b*self.y[-1] - self.a*self.c)/(self.a**2 + self.b**2)
        self.y_end = (self.a**2*self.y[-1] - self.a*self.b*self.x[-1] - self.b*self.c)/(self.a**2 + self.b**2)

        self.length = dist(self.x_start, self.y_start, self.x_end, self.y_end)

    def p2l_dist(self, rm, phim): # measures point to line distance
        xm, ym = to_cartesian(rm, phim)
        dist = np.abs(self.a*xm + self.b*ym + self.c)/np.sqrt(self.a**2 + self.b**2)
        return dist

    def plot_line(self):
        plt.plot(self.x, self.y, '.')
        plt.plot([self.x_start, self.x_end], [self.y_start, self.y_end], '-')
        plt.axis("equal")
        plt.show()

    def to_global(self, phit, xt, yt):
        """
        Given state {xt, yt, phit} of the vehicle, transform the line to the global frame.
        """
        if self.frame == "global":
            print("Warning: transforming line from global frame!")

        pstart = Point(self.x_start, self.y_start)
        pend = Point(self.x_end, self.y_end)

        pstart.to_global(phit, xt, yt)
        pend.to_global(phit, xt, yt)

        self.fit_line([pstart.x, pend.x], [pstart.y, pend.y])
        self.x_start, self.y_start = pstart.x, pstart.y
        self.x_end, self.y_end = pend.x, pend.y

        self.frame = "global"

    def __repr__(self):
        self.plot_line()
        return "Line object, start index: {0}, end index: {1}".format(self.start, self.end)

class Seed(Line):

    obj_type = "Seed"

    def __init__(self, i, j, r, phi):
        self.start = i
        self.end = j
        self.r = r
        self.phi = phi
        self.x, self.y = to_cartesian(self.r, self.phi)
        self.fit_line(self.x, self.y)

    def to_line(self):
        return Line(self.start, self.end, self.r, self.phi)

    def p2p_dist(self, rm, phim): # measures point to predicted point distance
        xm, ym = to_cartesian(rm, phim) # measured point
        xp = -self.c*np.cos(phim)/(self.a*np.cos(phim) + self.b*np.sin(phim)) # predicted point
        yp = -self.c*np.sin(phim)/(self.a*np.cos(phim) + self.b*np.sin(phim)) # predicted point
        dist = np.sqrt((xm - xp)**2 + (ym - yp)**2)
        return dist


def update_state(r0, r1, x0, y0, head0):
    s0 = ScanFeature(r0)
    s1 = ScanFeature(r1)

    correspondences = 0
    h_change = 0
    for line1 in s1.lines:
        for line0 in s0.lines: # this is changed to lines in global frame(?)
            if (dist(line0.x_start, line0.y_start, line1.x_start, line1.y_start) < corspd and
                dist(line0.x_end, line0.y_end, line1.x_end, line1.y_end) < corspd) \
            or (dist(line0.x_start, line0.y_start, line1.x_end, line1.y_end) < corspd and
                dist(line0.x_end, line0.y_end, line1.x_start, line1.y_start) < corspd):

                correspondences += 1
                h_change += wraptohalfpi(line1.heading - line0.heading)
                break
    try:
        d_head = -h_change/correspondences
        if abs(d_head) > 0.2:
            d_head = 0
    except:
        d_head = 0

    head = wraptopi(head0 + d_head)

    s1t = ScanFeature(r1, phi=np.linspace(-2*np.pi/3+d_head, 2*np.pi/3+d_head, len(r1)))
    correspondences = 0
    x_change = 0
    y_change = 0
    for kp1 in s1t.kpoints:
        for kp0 in s0.kpoints:
            if dist(kp0.x, kp0.y, kp1.x, kp1.y) < corspd:
                correspondences += 1
                x_change += (kp1.x - kp0.x)
                y_change += (kp1.y - kp0.y)
                break
    try:
        d_x = -x_change/correspondences
        d_y = -y_change/correspondences
        if abs(d_x) > 0.02 or abs(d_y) > 0.02: # scan error
            d_x, d_y = 0, 0
    except:
        d_x, d_y = 0, 0

    x = x0 + np.cos(head0) * d_x - np.sin(head0) * d_y
    y = y0 + np.sin(head0) * d_x + np.cos(head0) * d_y

    return x, y, head


def update_map(gf, r, x_st, y_st, head_st):
    s = ScanFeature(r)
    s.to_global(head_st, x_st, y_st)

    aligned = True
    for line in s.lines:
        match = False
        for gfline in gf.lines: # this is changed to lines in global frame(?)
            if abs(gfline.a - line.a) < 0.2 and abs(gfline.b - line.b) < 0.2 and abs(gfline.c - line.c) < 0.2:
                match = True
                if gfline.b == -1: # if line is horizontal:
                    gfline.a = (gfline.a*gfline.it + line.a) / (gfline.it+1)
                    gfline.c = (gfline.c*gfline.it + line.c) / (gfline.it+1)

                    if gfline.x_end > gfline.x_start:
                        gfline.x_end = max(gfline.x_end, line.x_start, line.x_end)
                        gfline.x_start = min(gfline.x_start, line.x_start, line.x_end)
                    else:
                        gfline.x_end = min(gfline.x_end, line.x_start, line.x_end)
                        gfline.x_start = max(gfline.x_start, line.x_start, line.x_end)

                    gfline.y_end = gfline.a * gfline.x_end + gfline.c
                    gfline.y_start = gfline.a * gfline.x_start + gfline.c

                elif gfline.a == -1: # if line is vertical
                    gfline.b = (gfline.b*gfline.it + line.b) / (gfline.it+1)
                    gfline.c = (gfline.c*gfline.it + line.c) / (gfline.it+1)

                    if gfline.y_end > gfline.y_start:
                        gfline.y_end = max(gfline.y_end, line.y_start, line.y_end)
                        gfline.y_start = min(gfline.y_start, line.y_start, line.y_end)
                    else:
                        gfline.y_end = min(gfline.y_end, line.y_start, line.y_end)
                        gfline.y_start = max(gfline.y_start, line.y_start, line.y_end)

                    gfline.x_end = gfline.b * gfline.y_end + gfline.c
                    gfline.x_start = gfline.b * gfline.y_start + gfline.c

                gfline.it += 1

                break

        if not match:
            line.it = 1
            gf.lines.append(line)
