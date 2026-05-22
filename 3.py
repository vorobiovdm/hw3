import cv2
import numpy as np

detector = cv2.QRCodeDetector()


def warp_qr(frame, bbox):
    if bbox is None:
        return None

    pts1 = bbox[0].astype(np.float32)

    size = 400
    pts2 = np.float32([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame, M, (size, size))

    return warped


def run():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        data_raw, bbox_raw, _ = detector.detectAndDecode(frame)
        raw_ok = bool(data_raw)


        retval, bbox = detector.detect(frame)


        data_warp = None
        warp_ok = False

        if retval and bbox is not None:
            warped = warp_qr(frame, bbox)

            if warped is not None:
                data_warp, _, _ = detector.detectAndDecode(warped)
                warp_ok = bool(data_warp)
        else:
            warped = None


        if bbox_raw is not None:
            b = bbox_raw[0].astype(int)
            for i in range(4):
                cv2.line(frame,
                         tuple(b[i]),
                         tuple(b[(i + 1) % 4]),
                         (0, 255, 0), 2)


        cv2.putText(frame, f"RAW: {data_raw if data_raw else 'None'}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"WARP: {data_warp if data_warp else 'None'}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"RAW_OK={raw_ok} | WARP_OK={warp_ok}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)


        cv2.imshow("RAW vs WARP", frame)

        if warped is not None:
            cv2.imshow("WARPED", warped)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
