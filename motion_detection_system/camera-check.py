import cv2


def cam_test() -> None:
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():  
        print("failed to open cam")
    else:
        print('cam opened on port {}'.format(0))

        for i in range(10 ** 10):
            success, cv_frame = cap.read()
            if not success:
                print('failed to capture frame on iter {}'.format(i))
                break
            cv2.imshow('Input', cv_frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    cam_test()
