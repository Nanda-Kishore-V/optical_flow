def main():
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.resize(gray, (240, 320))
        #call function here
        cv2.imshow('frame',edge_map)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos_dir = os.fsencode('videos')
    for file in os.listdir(test_dir):
        test_video = os.fsdecode(file)
        cannyEdgeVideo('videos/' + test_video)


if __name__ == "__main__":
    main()