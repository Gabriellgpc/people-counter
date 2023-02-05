class TrackableObject:
    def __init__(self, objectID, centroid):
        # object Id and its list of centroids
        self.objectID = objectID
        self.centroids = [centroid]
        # a boolean indicating whether this object has been counted or not
        self.counted = False