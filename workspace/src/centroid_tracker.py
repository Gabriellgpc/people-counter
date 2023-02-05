from scipy.spatial import distance
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        # A counter used to assign unique IDs to each object
        self.nextObjectID = 0
        # A dictionary that utilizes the object ID as the key and the centroid
        self.objects = OrderedDict()
        # Maintains number of consecutive frames (value) a particular object ID (key) has been marked as “lost”
        self.disappeared = OrderedDict()
        # The number of consecutive frames an object is allowed to be marked
        # as “lost/disappeared” until we deregister the object.
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        """
            When registering an object we use the next available object ID
            to store the centroid
        """
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """
            To deregister and object ID we delete
            the object ID from both of our respective dictionaries
        """
        del self.objects[objectID]
        del self.disappeared[objectID]
    def update(self, rects):
        """
            The format of the rects parameter is assumed
            to be a tuple with this structure: (startX, startY, endX, endY) .
        """
        # check if the list of rects is empty
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.register(objectID)
            return self.objects

        # convert the list of rects to a list of their respective centroids
        inputCentroids = np.zeros( shape = (len(rects),2), dtype='int')
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int( (startX + endX) / 2.0 )
            cY = int( (startY + endY) / 2.0 )
            inputCentroids[i] = (cX, cY)

        # if we do not have any tracked object yet,
        # create new ID for each one of those centroids and start
        # tracking them
        if len(self.objects) == 0:
            for i in range( len(inputCentroids) ):
                self.register( inputCentroids[i] )
        # otherwise, we need to try to match the input centroids
        # to existing objects centroids that minimize the euclidean distance
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pairof object
            # centroids and input centroids, respectively
            D = distance.cdist( np.array(objectCentroids), inputCentroids )

            # steps to do the matching
            # (1) find the smallest value in each row and
            # (2) sort the row indexes based on their minimum values
            rows = D.min(axis=1).argsort()

            # similar process in the columns by findind
            # the smallest value in each column and sorting
            # using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # to determine whether we need to update, register
            # or deregister an object we need to keep track of
            # which of the rows and columns we have already examined
            usedRows = set()
            usedCols = set()

            for (row,col) in zip(rows,cols):
                # if we have already examined either the row or columns
                # value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # for the row and columns index we have not yet examined
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            # in case the number of object centroids is equal or greater than the number of input
            # centroids, we need to check and see if some of these
            # objects have disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check if the number of consecutive frames
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregistered[objectID]
            # otherwise, if the number of input centroids if greater
            # than the number of existing object centroids, we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        # return the set of trackable objects
        return self.objects