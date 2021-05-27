class gestureDetector:
    def __init__(self,name,detector,confidence_threshold,image_height,image_width):
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.detector = detector
        self.image_height = image_height
        self.image_width = image_width
        print("Initialized {} with confidence threshold: {}".format(self.name,self.confidence_threshold))

    
    def detect_image(self,imageInput):
        return bool , str , int , int , int