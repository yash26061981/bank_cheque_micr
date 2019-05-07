
from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
import shutil


class BankChequeMICR:
    def __init__(self):
        self.result_dir = 'result/'
        self.reference_micr = 'reference/micr_e13b_reference.png'
        self.keep_bottom_part = 0.2
        self.rsz_h, self.rsz_w = 36, 36
        self.crop_offset = 5
        self.draw_offset = 10
        self.min_pass_w, self.min_pass_h = 0.02, 0.15
        # initialize the list of reference character names, in the same
        # order as they appear in the reference image where the digits
        # their names and:
        # T = Transit (delimit bank branch routing transit)
        # U = On-us (delimit customer account number)
        # A = Amount (delimit transaction amount)
        # D = Dash (delimit parts of numbers, such as routing or account)
        self.charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]
        self.display = True
        self.draw_predicted = True
        self.put_predicted_text = True

    def remove_directory_contents(self, path):
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    # To show the image.
    def show_image(self, in_image):
        cv2.imshow('image', in_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def extract_digits_and_symbols(self, image, charCnts, minW=5, minH=15): # was 5 and 15
        char_iter = charCnts.__iter__()
        rois = []
        locs = []
        while True:
            try:
                c = next(char_iter)
                (cX, cY, cW, cH) = cv2.boundingRect(c)
                if cW >= minW and cH >= minH:
                    # extract the ROI
                    roi = image[cY:cY + cH, cX:cX + cW]
                    rois.append(roi)
                    locs.append((cX, cY, cX + cW, cY + cH))

                else:
                    parts = [c, next(char_iter), next(char_iter)]
                    (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
                        -np.inf)

                    # loop over the parts
                    for p in parts:
                        (pX, pY, pW, pH) = cv2.boundingRect(p)
                        sXA = min(sXA, pX)
                        sYA = min(sYA, pY)
                        sXB = max(sXB, pX + pW)
                        sYB = max(sYB, pY + pH)

                    # extract the ROI
                    roi = image[sYA:sYB, sXA:sXB]
                    rois.append(roi)
                    locs.append((sXA, sYA, sXB, sYB))

            except StopIteration:
                break
        return rois, locs

    def process_reference(self):
        ref = cv2.imread(self.reference_micr)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = imutils.resize(ref, width=400)
        ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        ref_cnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        ref_cnts = ref_cnts[0] if imutils.is_cv2() else ref_cnts[1]
        ref_cnts = contours.sort_contours(ref_cnts, method="left-to-right")[0]

        ref_roi = self.extract_digits_and_symbols(ref, ref_cnts, minW=10, minH=20)[0]
        chars = {}
        for (name, roi) in zip(self.charNames, ref_roi):
            roi = cv2.resize(roi, (self.rsz_h, self.rsz_w))
            chars[name] = roi

        return chars

    def get_bottom_cropped_image(self, in_image):
        (h, w,) = in_image.shape[:2]
        delta = int(h - (h * self.keep_bottom_part))
        return in_image[delta:h, 0:w], delta

    def get_processed_bmp(self, in_image):
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
        bottom_cropped, delta = self.get_bottom_cropped_image(in_image)

        gray = cv2.cvtColor(bottom_cropped, cv2.COLOR_BGR2GRAY)
        blackhat_image = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        scharr_grad_x = cv2.Sobel(blackhat_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        scharr_grad_x = np.absolute(scharr_grad_x)
        (minVal, maxVal) = (np.min(scharr_grad_x), np.max(scharr_grad_x))
        scharr_grad_x = (255 * ((scharr_grad_x - minVal) / (maxVal - minVal)))
        scharr_grad_x = scharr_grad_x.astype("uint8")

        scharr_grad_x_closed = cv2.morphologyEx(scharr_grad_x, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.threshold(scharr_grad_x_closed, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = clear_border(thresh)
        return gray, thresh, delta

    def get_grouped_contours(self, in_bmp):
        (oh, ow,) = in_bmp.shape[:2]
        w_th = (int)(ow*self.min_pass_w)
        h_th = (int)(oh*self.min_pass_h)
        group_cnts = cv2.findContours(in_bmp, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        group_cnts = group_cnts[0] if imutils.is_cv2() else group_cnts[1]
        group_locs = []
        for (i, c) in enumerate(group_cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            if w >= w_th and h >= h_th: # was 50 and 15
                group_locs.append((x, y, w, h))

        # sort the digit locations from left-to-right
        group_locs = sorted(group_locs, key=lambda x: x[0])
        return group_locs

    def get_group_roi_from_gray_scale(self, gray, gx, gy, gw, gh):
        group = gray[gy - self.crop_offset:gy + gh + self.crop_offset, gx - self.crop_offset:gx + gw + self.crop_offset]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return group

    def process_cheque(self, in_image):
        chars = self.process_reference()
        output = []
        image = cv2.imread(in_image)
        image = imutils.resize(image, width=1024)
        draw_image = image.copy()
        gray, thresh, delta = self.get_processed_bmp(image)
        group_locs = self.get_grouped_contours(thresh)

        for (gX, gY, gW, gH) in group_locs:
            group_output = []
            group = self.get_group_roi_from_gray_scale(gray, gX, gY, gW, gH)
            '''
            if self.display:
                self.show_image(group)
            '''
            char_cnts = cv2.findContours(group, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            char_cnts = char_cnts[0] if imutils.is_cv2() else char_cnts[1]
            if len(char_cnts)==0:
                continue
            char_cnts = contours.sort_contours(char_cnts, method="left-to-right")[0]
            rois = self.extract_digits_and_symbols(group, char_cnts)[0]

            for roi in rois:
                scores = []
                roi = cv2.resize(roi, (self.rsz_h, self.rsz_w))

                for charName in self.charNames:
                    # apply correlation-based template matching, take the
                    # score, and update the scores list
                    result = cv2.matchTemplate(roi, chars[charName],
                                               cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                # the classification for the character ROI will be the
                # reference character name with the *largest* template
                # matching score
                group_output.append(self.charNames[np.argmax(scores)])

            if self.draw_predicted:
                # draw (padded) bounding box surrounding the group along with
                # the OCR output of the group
                cv2.rectangle(draw_image, (gX - self.draw_offset, gY + delta - self.draw_offset),
                              (gX + gW + self.draw_offset, gY + gY + delta), (0, 0, 255), 2)
                if self.put_predicted_text:
                    cv2.putText(draw_image, "".join(group_output),
                                (gX - 10, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.95, (0, 0, 255), 3)

            # add the group output to the overall check OCR output
            output.append("".join(group_output))

        # display the output check OCR information to the screen
        print("Check OCR: {}".format(" ".join(output)))
        if self.display:
            self.show_image(draw_image)
        self.save_images(in_image, draw_image, output)

    def save_images(self, name, marked_image, ocr_op):
        output_link_file = os.path.join(self.result_dir, 'ocr.txt')
        file_ptr = open(output_link_file, 'a')
        words = name.split("\\")
        patch_words = words[-1].split(".")
        new_name = os.path.join(self.result_dir, (str(patch_words[0]) + '_ocr.png'))
        cv2.imwrite(new_name, marked_image)
        data_to_write = (words[-1], " ".join(ocr_op))
        file_ptr.write(", ".join(str(v) for v in data_to_write))
        file_ptr.write('\n')
        file_ptr.close()

    def get_cheque_samples_from_dir(self, srcdir):
        for filename in os.listdir(srcdir):
            sample_name = os.path.join(srcdir, filename)
            print('Processing: ', sample_name)
            self.process_cheque(sample_name)


if __name__ == "__main__":
    input_image = 'data/example_check.png'
    micr_extractor = BankChequeMICR()
    micr_extractor.remove_directory_contents(micr_extractor.result_dir)
    #micr_extractor.process_cheque(input_image)
    src_dir = 'sample_data'
    micr_extractor.get_cheque_samples_from_dir(src_dir)
    print('Done')