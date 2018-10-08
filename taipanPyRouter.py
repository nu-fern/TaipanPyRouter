import numpy as np
import json
import plotter
import pylab as plt 
import constants as const
from scipy.spatial import distance 
from scipy.optimize import linear_sum_assignment
from shapely.geometry.linestring import LineString
import argparse
import logging
import os.path
import itertools
import os
import dill
# ---- NU - NEW EDITS --------#
import pandas as pd


#avoid warning from NaNs and infs
np.warnings.filterwarnings('ignore')


bugStatusDict = {'ONLINE':0,
                 'BROKEN':1,
                 'INACTIVE':2,
                 'NOT_PRESENT':3}


#Dictionary of bug types. None defaults to SCIENCE 
bugTypeDict = {'GUIDE' : 0,
               'SKY' : 1,
               'SCIENCE':2,
               'None' : 3} 

bugRoutingDict = {'DIRECT' : 0,
                  'SEQUENCE' : 1,
                  'UNROUTABLE' : 2,
                  'NO_TARGET' : 3,
                  'OUTSIDE_PR' : 4,
                  'OUTSIDE_GFP' : 5,
                  'CRASHING' : 6,
                  'NONE' : 7}

logger = logging.getLogger('TaipaPyRouter')

def loadParkPosJSON( filename = 'locationProperties.json', folder = '.'):   
    '''
    Reads the park position file
    
    Args:
        filename (str) : The name of the input file
        folder (str) : The location of the input file
        
    Returns:
        parkPos (np.ndarray) : 2D - [[x],[y]] parked positions coordinates indexed by LemoId-1 
        bugStatus (np.ndarray) : 1D [status] bugStatus indexed by LemoId-1
        bugTypes (np.ndarray) : 1D [types] bugTypes indexed by LemoId-1
    
    Note:
        Currently hardcoded array size to 309 bugs
        This function should read from the database    
    '''

    parkPos = np.ones((309,2)) * np.nan
    bugStatus = np.zeros(309)
    bugTypes = np.zeros(309)
    bugRouting = np.zeros(309)

    #-------------------------------- NU EDITS ---------------------------------#
    bugDeployInfo = pd.read_excel('NumberingMatrix_17May2018.xlsx', sheetname='SOCKETS')
    deployedLEMO = list(bugDeployInfo[bugDeployInfo['160 Plugs'] == 1].get('LEMO'))
    
    with open(filename, 'r') as f:
        dataAll = json.loads(f.read())


    for data in dataAll['locationProperties']: 
        bugIdx = data['lemoID']-1 #idx = bugID-1
        park = data['homePosition']
        x = park['xMicrons']
        y = park['yMicrons'] 
        parkPos[bugIdx,:] = x,y 

        #populate bugTypes array
        bugTypes[bugIdx] = bugTypeDict[str(data['sbType'])]
        
        #populate bugStatus array
        #Filter out non-deployed LEMOS -  NU - EDITS
            # non-deployed -> NOT_PRESENT - 3
            # deployed     -> INACTIVE    - 2    %  ( now ONLINE - 0 for testing)
        if (str(data['lemoID']).zfill(3) in deployedLEMO):
            bugStatus[bugIdx] = 0 # bugStatusDict[str(data['sbState'])] - NU- EDITED
        else:
            bugStatus[bugIdx] = 3

        #populate bugRouting array
        bugRouting[bugIdx] = bugRoutingDict['NONE']

    return parkPos, bugStatus, bugTypes, bugRouting



def openS2JSONTile(fileName, folder = './jsonTiles_s2'):
    '''
    Reads the JSON file containing the target information (S2)
    
    Args:
        filename (str) : The name of the input file
        folder (str) : The location of the input file
        
    Returns:
        bugsXY (np.ndarray) : Bugs requested position [[x,y]] indexed by LemoId-1 
        
    Note:
        - Currently hardcoded array size to 309,2
    '''

    with open(folder + '/' + fileName, 'r') as f:
        data = json.loads(f.read())


    bugsTargetXY = np.ones((309,2)) * np.nan
    bugTargetTypes = np.ones(309) * np.nan

    reqGuideBugsIdx = []
    reqScienceBugsIdx = []
    
    #Collects all found bug data
    for i,thisData in enumerate([data['guideStars'], data['sky'], data['targets']]):

        for thisBug in thisData:
            if i == 0:
                reqGuideBugsIdx.append(thisBug['sbID']-1)
            else:
                reqScienceBugsIdx.append(thisBug['sbID']-1)
            
            bugsTargetXY[thisBug['sbID']-1,:] = [thisBug['xMicrons'], thisBug['yMicrons']]    
            
            bugTargetTypes[thisBug['sbID']-1] = i 

    return bugsTargetXY, bugTargetTypes


def initialiseTickArray(parkPos, bugsTargetXY):
    '''
    Creates a new tick array with 2 only ticks (direct path)
    
    Args:
        parkPos: array with the starting points
        bugsTargetXY: array with the end points
        
    Returns:
        tickArray: 3D np.array [lemoID-1, tick, coords] 


    '''
    
    tickArray = np.ones((309,2,2))*np.nan
    tickArray[:,0,:] = parkPos
    tickArray[:,1,:] = bugsTargetXY
         
    return tickArray

def checkValidGFPandCrash(bugsTargetXY, bugTargetTypes):
    '''
    Checks that targets are:
        - inside GFP
        - far enough from eachother
        
    Removes 
    
    Args:
        bugsTargetXY: array with the target points
        bugsTargetStatus: array with the end points
                
    '''
    logger.info('Initial Validation of Targets')
    
    #The distance from the centre. Should be inside GFP
    R = ((bugsTargetXY[:,0])**2+(bugsTargetXY[:,1])**2)**0.5
    filter = R > (const.GFP_RADIUS-const.BUG_RADIUS)
    if np.sum(filter)>0:
        strMsg = np.sum(filter), ' bugs outside GFP.'
        print(strMsg)
        logger.warning(strMsg)
        bugsTargetXY[filter] = np.nan
        bugTargetTypes[filter] = np.nan

    #Check for target crash
    R2d = distance.cdist(bugsTargetXY, bugsTargetXY, 'euclidean')
    tooCloseList = np.where(R2d<(2*const.BUG_RADIUS))
    toRemove = np.unique(tooCloseList[0][tooCloseList[0]>tooCloseList[1]])

    bugsTargetXY[toRemove] = np.nan
    bugTargetTypes[toRemove] = np.nan
    
    if toRemove.shape[0]:    
        strMsg = 'Bugs', toRemove, ' were removed by having another target too close.'
        print (strMsg)
        logger.warning(strMsg)



def checkValidTargetsPR(parkPos, bugsTargetXY, bugStatus, bugRouting):
    '''
    Checks that targets are:
        - inside patrol radius
        - far from static bugs
        
    Updates bugStatus accordingly
    
    Args:
        parkPos: array with the starting points
        bugsTargetXY: array with the end points
                
    '''
    logger.info('Validating Target Patrol Radius')
    
    #The distance travelled by each bug. Should be inside patrol radius
    R = ((parkPos[:,0]-bugsTargetXY[:,0])**2+(parkPos[:,1]-bugsTargetXY[:,1])**2)**0.5
    filter = R>const.PATROL_RADIUS
    if np.sum(filter)>0:
        strMsg = np.sum(filter), ' bugs outside patrol radius.'
        print(strMsg)
        logger.warning(strMsg)
        bugStatus[filter] = bugStatusDict['INACTIVE']
        bugRouting[filter] = bugRoutingDict['OUTSIDE_PR']

     
    #Check for target crash
    bugsTargetXYTemp = bugsTargetXY.copy()
    bugsTargetXYTemp[bugStatus==bugStatusDict['INACTIVE']] = parkPos[bugStatus== bugStatusDict['INACTIVE']] #update with invalid targets so far
    
    #iteratively mark as invalid crashing bugs
    notFinished = True
    totTooClose = 0
    while(notFinished==True):
        nChanges = 0       
        R2d = distance.cdist(bugsTargetXYTemp, bugsTargetXYTemp, 'euclidean')
        tooCloseList = np.where(R2d<(2*const.BUG_RADIUS))
        for i,j in zip(tooCloseList[0],tooCloseList[1]):
            if i<j: 
                nChanges+=1
                totTooClose+=2
                bugsTargetXYTemp[i] = parkPos[i]
                bugStatus[i]= bugStatusDict['INACTIVE'] 
                bugRouting[i] = bugRoutingDict['CRASHING']
                bugsTargetXYTemp[j] = parkPos[j]
                bugStatus[j]= bugStatusDict['INACTIVE']
                bugRouting[j] = bugRoutingDict['CRASHING']
        if nChanges==0: notFinished = False
    
    if totTooClose>0:
        strMsg = totTooClose, ' bugs were too close to each other.'
        print(strMsg)
        logger.warning(strMsg)






def checkValidTargets(parkPos, bugsTargetXY, bugStatus, bugRouting):
    '''
    Checks that targets are:
        - inside patrol radius
        - far from static bugs
        
    Updates bugStatus accordingly
    
    Args:
        parkPos: array with the starting points
        bugsTargetXY: array with the end points
                
    '''
    logger.info('Validating Targets')
    
    #The distance travelled by each bug. Should be inside patrol radius
    R = ((parkPos[:,0]-bugsTargetXY[:,0])**2+(parkPos[:,1]-bugsTargetXY[:,1])**2)**0.5
    filter = R>const.PATROL_RADIUS
    if np.sum(filter)>0:
        strMsg = np.sum(filter), ' bugs outside patrol radius.'
        print(strMsg)
        logger.warning(strMsg)
        bugStatus[filter] = bugStatusDict['INACTIVE']
        bugRouting[filter] = bugRoutingDict['OUTSIDE_PR']

     
    #Check for target crash
    bugsTargetXYTemp = bugsTargetXY.copy()
    bugsTargetXYTemp[bugStatus==bugStatusDict['INACTIVE']] = parkPos[bugStatus== bugStatusDict['INACTIVE']] #update with invalid targets so far
    
    #iteratively mark as invalid crashing bugs
    notFinished = True
    totTooClose = 0
    while(notFinished==True):
        nChanges = 0       
        R2d = distance.cdist(bugsTargetXYTemp, bugsTargetXYTemp, 'euclidean')
        tooCloseList = np.where(R2d<(2*const.BUG_RADIUS))
        for i,j in zip(tooCloseList[0],tooCloseList[1]):
            if i<j: 
                nChanges+=1
                totTooClose+=2
                bugsTargetXYTemp[i] = parkPos[i]
                bugStatus[i]= bugStatusDict['INACTIVE'] 
                bugRouting[i] = bugRoutingDict['CRASHING']
                bugsTargetXYTemp[j] = parkPos[j]
                bugStatus[j]= bugStatusDict['INACTIVE']
                bugRouting[j] = bugRoutingDict['CRASHING']
        if nChanges==0: notFinished = False
    
    if totTooClose>0:
        strMsg = totTooClose, ' bugs were too close to each other.'
        print(strMsg)
        logger.warning(strMsg)




        
def findCrossingGroups(tickArray,bugStatus):
    '''
    Identify crossing groups within the direct paths
    
    Args:
        tickArray: array with the starting and ending points
        
    Returns:
        crossingBugs: np.ndarray Collection of pairs of crossing bugs.  

    '''
    
    activeFilter =  bugStatus!= bugStatusDict['NOT_PRESENT']
    crossingBugs = []
    
    for thisBugIdx, thisBug in zip(np.arange(tickArray.shape[0])[activeFilter],tickArray[activeFilter]):
        path1 = LineString(thisBug)
        
        for otherBugIdx, otherBug in zip(np.arange(tickArray.shape[0])[activeFilter],tickArray[activeFilter]):
    
            if thisBugIdx>otherBugIdx:
                path2 = LineString(otherBug)
                dist = path1.distance(path2)

                if dist < 2*const.CORRIDOR_HALF_WIDTH: 
                    crossingBugs.append([thisBugIdx,otherBugIdx])

    crossingBugs = np.array(crossingBugs)
    
    return crossingBugs


def consolidateCGroups(crossingBugs, tickArray):
    '''
    Takes all pairs that collide and groups them assigning a unique ID.
    
    Args:
        crossingBugs: np.ndarray Collection of pairs of crossing bugs.  
        tickArray: np.ndarray [lemoID-1, tick, coords] 
             
    '''
     
    CGroups = np.zeros(tickArray.shape[0])   
      
    for thisPair in crossingBugs:
        
        newGroupId = np.max(CGroups)+1
        
        bug1, bug2 = thisPair
        
        if CGroups[bug1]>0:
            bug1OldGroup = CGroups[bug1]
            CGroups[CGroups==bug1OldGroup] = newGroupId
        CGroups[bug1] = newGroupId
        
        if CGroups[bug2]>0:
            bug2OldGroup = CGroups[bug2]
            CGroups[CGroups==bug2OldGroup] = newGroupId
        CGroups[bug2] = newGroupId
        
    return CGroups


def optimiseAllocation(parkPos, bugStatus, bugTypes, bugsTargetXY, bugTargetTypes):
    '''
    Minimise the cost matrix of distances between a set of parked positions and set of targets.
     
    - It assigns the best target positions based on minimum combined distance
    - Only the positions in parkPos that can be allocated have actual values, the rest of the values are NaNs. 
    - This process allows the code to segment the allocation by fibre type

    Args:
        parkPos (np.ndarray) : Park position for all starbugs
        bugsTargetXY (np.ndarray) : Target positions to be allocated
     
    Returns:
        newTargetsAlloc (np.ndarray) : Array of same shape of parkPos with the new allocations
        
    '''
    
    #initialise output array
    newTargetsAlloc = np.ones((bugStatus.shape[0], 2))*np.nan
    
    #Availability filter for all bugs
    availableBugsFilter = (bugStatus==bugStatusDict['ONLINE'])



    '''
    GUIDE bugs
    '''
    
    # Create a list of the available GUIDE bugs parked positions (sources)
    availableGuideBugsFilter = ((bugTypes==bugTypeDict['GUIDE']) & availableBugsFilter)
    sourcesIdx = np.arange(bugStatus.shape[0])[availableGuideBugsFilter]
    sources = parkPos[availableGuideBugsFilter]   

     #Create a list of targets for GUIDE bugs (targets)
    tempBugsTargetXY = bugsTargetXY[bugTargetTypes==bugTypeDict['GUIDE']]
    targets = tempBugsTargetXY


    #Debug stuff
#     sources = sources[6:7,:]   
#     sources = np.delete(sources, 1,0)
#     targets = np.delete(targets, 1,0)
#     targets = targets[3:4,:]

   
    #create cost matrix
    C = distance.cdist(sources, targets)

    filter = C>const.PATROL_RADIUS
    C[filter]=C[filter]+np.max(C)           # Why the addition of max value ?
    
    #assign based on min cost
    rows, cols = linear_sum_assignment(C)
    
    #transfer first batch
    newTargetsAlloc[sourcesIdx[rows]] = targets[cols]
    
    
    
    '''
    SCIENCE and SKY bugs
    '''   
    
    #Create a list of available SCIENCE and SKY bugs park positions (sources)
    ScienceAndSkyFibresFilter = ((bugTypes==bugTypeDict['SCIENCE']) | (bugTypes==bugTypeDict['SKY']))
    availableScienceAndSkyFibresFilter = (ScienceAndSkyFibresFilter & availableBugsFilter)
    sourcesIdx = np.arange(bugStatus.shape[0])[availableScienceAndSkyFibresFilter]
    sources = parkPos[availableScienceAndSkyFibresFilter]   

    
    #Create a list of targets for SCIENCE and SKY bugs (targets)
    ScienceAndSkyTargetFilter = ((bugTargetTypes==bugTypeDict['SCIENCE']) | (bugTargetTypes==bugTypeDict['SKY']))
    tempBugsTargetXY = bugsTargetXY[ScienceAndSkyTargetFilter]
    targets = tempBugsTargetXY

    
    C = distance.cdist(sources, targets)

    filter = C>const.PATROL_RADIUS
    C[filter]=C[filter]+np.max(C)
    
    #assign based on min cost
    rows, cols = linear_sum_assignment(C)
    
    #transfer first batch
    newTargetsAlloc[sourcesIdx[rows]] = targets[cols]
    
    return newTargetsAlloc


class CGroupSolver:
    '''
    Class to hold the crossing group solving funcitons.
    '''
    def __init__(self,CGroup,tickArray,bugStatus):
        
        self.CGroup = CGroup
        self.tickArray = tickArray
        self.bugStatus = bugStatus
        self.eSegments = self.constructESegments(tickArray[CGroup,:2,:])
        self.C = self.constructCMatrix()
    
    def constructESegments(self, XYs):
        '''
        Constructs a list of all ESegments in the CGroup. 
        
        Args:
            XYs (np.ndarray): List of coordinates of the end points of each ESegment
            
        Returns:
            list: List of ESegments
        '''

        eSegments = []
        for s,e in XYs:
            
            thisESegment = ESegment([s,e])
            eSegments.append(thisESegment)
        

        return eSegments
  

    def calcCoeffs(self, A, B):
        '''
        Calculates the collision coefficients.
        
        - Given 2 ESegments, it returns the coefficients of the collision points.
        - If the ESegments are parallel, it returns NaNs
        
        Args:
            A (ESegment) : First segment to compare.
            B (ESegment) : Second segment to compare.
        
        Returns:
            Tuple: Pair of coefficients
        '''
        
        aNumerator = ((B.Xe-B.Xs)*(A.Ys-B.Ys))-((B.Ye-B.Ys)*(A.Xs-B.Xs))
        bNumerator = ((A.Xe-A.Xs)*(A.Ys-B.Ys))-((A.Ye-A.Ys)*(A.Xs-B.Xs))
        denominator = ((B.Ye-B.Ys)*(A.Xe-A.Xs))-((B.Xe-B.Xs)*(A.Ye-A.Ys))

        if denominator==0:
            coeffA = np.nan
            coeffB = np.nan
        else:
            coeffA = aNumerator/denominator
            coeffB = bNumerator/denominator

        return coeffA, coeffB

    
    def constructCMatrix(self):
        '''
        Creates a distance matrix for all segments in the crossing group
        '''
        nESegments = len(self.eSegments)
        C = np.ones((nESegments,nESegments))*1e9

        for i,thisESegment in enumerate(self.eSegments):
            for j,otherESegment in enumerate(self.eSegments):
                if i>j:
                    C[i,j], C[j,i] =  self.calcCoeffs(thisESegment, otherESegment)
        return C
    
    def findMovingSequence(self):
        '''
        Analyses the cost matrix to find the sequence of motion that doesn't crash.
        '''

        a = self.C.copy()
        result = []

        for i in range(self.CGroup.shape[0]):

            minR = np.where(a == np.nanmin(a))[0][0]
            result.append(self.CGroup[minR])
            a[minR,:] = 1e9
      
        return np.array(result)

    def findMovingSequenceBF(self):
        booCollisions = True

        #contains the list of possible allocations
        allocArray = np.array(list(itertools.permutations(range(self.CGroup.shape[0]))))
        for thisAlloc in allocArray:
            
            for i in range(1,self.tickArray.shape[1]):
                self.tickArray[self.CGroup,i,:] = self.tickArray[self.CGroup,-1,:][thisAlloc]
            
            for thisSeq in np.array(list(itertools.permutations(self.CGroup[thisAlloc]))):
            
                
                #rebuild tick array with the new sequence
                tempTickArray = shiftTickArray(thisSeq, self.tickArray)
            
                if checkForCollisions(tempTickArray[self.CGroup,:,:], self.bugStatus[self.CGroup], False):
                    booCollisions = True
                     
                else:
                    booCollisions = False
                    break
                
            #To leave the second for loop
            if booCollisions == False: break
            
        return booCollisions, thisSeq, tempTickArray
        
        
        
        
class ESegment(LineString):
    '''
    Class to extend the existing Linestring class into E(xtended)Segments.
    
    ESegments provide extra elements that apply to route solving
    
    '''
    def __init__(self, XYs):
        
        LineString.__init__(self, XYs)
        
        self.Xs = XYs[0][0]
        self.Xe = XYs[1][0]
        self.dx = self.Xe-self.Xs
        
        self.Ys = XYs[0][1]
        self.Ye = XYs[1][1]
        self.dy = self.Ye-self.Ys


    def pointFromCoeff(self, coeff):
        '''
        Calculates a projected point from a given coefficient. 
        
        The resulting point is in the position coeff*length, where length is 
        the length of the ESegment. 
        
        Args:
            coeff (float) : Distance to the requested point in ESegment lengths. Can be negative.
            
        Return:
            Tuple: Position of the point. 
        '''        
        return self.Xs+self.dx*coeff, self.Ys+self.dy*coeff 



def shiftTickArray(movSeq, tickArray):
    '''
    Given a sequence of bugIds, it creates the ticks to move bugs sequentially
    
    - It reshapes tickArray as needed. [nBugs, len(movSeq)+1, 2]
    - It cascades the motion of the corresponding bugs sequentially as per movSeq order
    - It completes preceding ticks with initial tick pos for movSeq bugs
    - It adds target position to all tick following motion
    
    
    Args:
        movSeq (np.ndarray) : List of bugIds to be moved in sequential order.
        tickArray (np.ndarray) : Tick array to be appended with sequential motion 
    
    Returns:
        np.ndarray: Shifted TickArray.
    
    '''
    
    initialNTicks = tickArray.shape[1]
    newNTicks = movSeq.shape[0]+1
    
    #Do we need to increase size? 
    if initialNTicks<newNTicks:
        newShape = np.array(tickArray.shape)
        newShape[1] = newNTicks
    else:
        newShape = tickArray.shape

    tempTickArray = np.zeros(newShape)
    #fill up existing data
    tempTickArray[:,:initialNTicks,:] = tickArray.copy()
    
    #Complete remaining columns with final position
    for i in range(initialNTicks,newNTicks):
        tempTickArray[:,i,:] = tempTickArray[:,initialNTicks-1,:]
    
    #Complete remaining columns with final position
    for i,thisBugId in enumerate(movSeq):
        if i>0:
            for j in range(i):
                tempTickArray[thisBugId,j+1,:] = tempTickArray[thisBugId,j,:]

    return tempTickArray


def checkForCollisions(tickArray, bugStatus, booPrint=True):
    '''
    Looks for collisions in the created tick array.
    
    - Steps through the tickArray 1 tick at the time.
    - Tries to re-create the crossing groups to look for crossings
    
    Args:
        tickArray (np.ndarray) : Array with the routing solution
        bugStatus (np.ndarray) : Status of the bugs
        
    Returns: 
        boolean : True if collisions found. 
    '''

    result = False
    for i in range(tickArray.shape[1]-1):
        
        #slice into single tick
        tempTickArray = tickArray[:,i:i+2,:]
        
        #Create crossing group array
        tempCGroups = findCrossingGroups(tempTickArray, bugStatus)    
        CGroups = consolidateCGroups(tempCGroups, tempTickArray)
        

        
        #output the CGroups for reference
        if booPrint: print 
        if booPrint: print('Collision Report tick',i,'->',i+1)
        
        for j in np.unique(CGroups):
            if j==0:
#                 result = False
                if booPrint: print('Number of bugs not colliding: ', np.sum(CGroups==j))
            else:
                result = True
                if booPrint: print('CGroup',int(j),':', np.sum(CGroups==j), 'collisions.' )
                
    return result
   


def writeOuputFile(S2FileName, S3FileName, tickArray):
    '''
    Writes an RTile (S3) from a tickArray
    
    Args:
        S2FileName (string) : Input XYTile
        S3FileName (string) : Output RTile
        tickArray (np.ndarray) : routing ticks array
    '''
    
    #Load S2
    with open('./jsonTiles_s2/'+S2FileName, 'r') as inFile:
        S2data = json.loads(inFile.read())

    
    #add tickArray data
    S3data = S2data
    S3data['schemaID']=3
    
    ticks={}    
    bugsWithoutNaNs = ~np.isnan(np.sum(tickArray, axis=1))[:,0]

    for thisTickIdx in range(tickArray.shape[1]):
        thisTickData = {}
        for thisBugIdx in np.arange(tickArray.shape[0])[bugsWithoutNaNs]:
            
            thisBugData = {}
            thisBugData['lemoID'] =  np.asscalar(thisBugIdx)+1
            thisBugData['xMicrons'] =  tickArray[thisBugIdx,thisTickIdx,0]
            thisBugData['yMicrons'] =  tickArray[thisBugIdx,thisTickIdx,1]
            
            thisTickData[str(thisBugIdx)] = thisBugData

        ticks['Tick ' + str(thisTickIdx)] = thisTickData
            
    S3data['routes'] = ticks

    #- NU - EDIT - Turn to key: value pairs
    s3map = [{'key': str(k), 'value': v} for k, v in S3data.items()]

    #write S3
    with open(S3FileName , 'w') as outFile:
        json.dump(s3map, outFile) #json.dump(S3data, outFile)-> translate from python2.7 to 3.6

    
def createWorkingFolder(base = '.'):
    '''
    Creates a folder to drop files using the next available name. 
    '''
    
    _, folders, _ = os.walk('.').__next__() #-----------------------------
    folders = np.array(folders)
    firsts = [w[:3] for w in folders]
    runs = np.array(firsts)=='run'
    runsUsed = np.array([w[4:] for w in folders[runs]]).astype(int)
    
    if runsUsed.shape[0]==0:
        folderName = 'run_1'
    else:
        folderName = 'run_'+ str(np.max(runsUsed)+1)

    return folderName
    


def doRoutes(args):
    
    #Initialise logging module 
    if logger.handlers==[]:
        hdlr = logging.FileHandler('temp.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(logging.DEBUG)
    logger.info('Module loading started')

    workingFolder = createWorkingFolder()
    if not os.path.isdir(workingFolder): os.mkdir(workingFolder)
    
    #parse input filename and sanitise
    S2FileName = args.f
    if S2FileName[-5:] != '.json': S2FileName += '.json'
    strMsg = 'Input filename is ' + S2FileName
    print(strMsg)
    logger.info(strMsg)       
    
    #parse output filename and sanitise
    if args.o==None:
        S3FileName = 'RTile_' + S2FileName
    else:
        S3FileName = args.o
    if S3FileName[-5:] != '.json': S3FileName += '.json'
    strMsg = 'Output filename is ' + S3FileName
    print(strMsg)
    logger.info(strMsg)       
        
        



    #Check input file exisits
    if os.path.isfile('./jsonTiles_s2/'+S2FileName): 
    
        #Load input files
        parkPos, bugStatus, bugTypes, bugRouting = loadParkPosJSON()
        
        
        #Hacks to simulate differnent bug configs*****************************
#         bugStatus = np.ones(309)*bugStatusDict['ONLINE'] #All online
        TOP=100
#         bugStatus[:TOP] = np.ones(TOP)*bugStatusDict['ONLINE'] #All online

#         bugStatus[:159] = np.ones(159)*bugStatusDict['ONLINE']
#         bugStatus[np.arange(228,234)]=0        
#         bugStatus = np.zeros(309) #Hack to make all bugs available ignoring input file info
        
        #plot park positions
        thisPlotName = 'ParkPos'
        ax =  plotter.plot_park_pos(parkPos, bugStatus, bugTypes)
        plt.savefig(workingFolder + '/' + thisPlotName)
        #dill.dump(ax, file(workingFolder + '/' + thisPlotName + '.dill', 'w')) ->> CARLOS - Python 2.7
        dill.dump(ax, open(workingFolder + '/' + thisPlotName + '.dill', 'wb')) # ->> NU - Python 3.6
        
        '''
        Output parkPos info
        '''
        strMsg = 'Found ' + str(parkPos.shape[0]) + ' park positions.'
        print(strMsg)
        logger.info(strMsg)

        unique, counts = np.unique(bugTypes, return_counts=True)
        keys = list(bugTypeDict.keys())
        values = list(bugTypeDict.values())
        strMsg = 'Bugs Types:'
        print(strMsg)
        logger.info(strMsg)
        for i,idx in enumerate(unique):
            strMsg = keys[values.index(idx)],counts[i]  
            print (strMsg)
            logger.info(strMsg)

        unique, counts = np.unique(bugStatus, return_counts=True)
        keys = list(bugStatusDict.keys())
        values = list(bugStatusDict.values())
        strMsg = 'Bugs Status:'
        print(strMsg)
        logger.info(strMsg)
        for i,idx in enumerate(unique):
            strMsg = keys[values.index(idx)], counts[i]  
            print(strMsg)
            logger.info(strMsg)

        
        
        
        
        #Load target positions and types
        bugsTargetXY, bugTargetTypes = openS2JSONTile(S2FileName)

        #plot Targets
        thisPlotName = 'targets'
        ax =  plotter.plot_targets(bugsTargetXY, bugTargetTypes)
        plt.savefig(workingFolder + '/' + thisPlotName)
        #dill.dump(ax, file(workingFolder + '/' + thisPlotName + '.dill', 'w')) ->> CARLOS - Python 2.7
        dill.dump(ax, open(workingFolder + '/' + thisPlotName + '.dill', 'wb')) # ->> NU - Python 3.6

 
        checkValidGFPandCrash(bugsTargetXY, bugTargetTypes)

        #plot Targets
        thisPlotName = 'targetsClean'
        ax =  plotter.plot_targets(bugsTargetXY, bugTargetTypes)
        plt.savefig(workingFolder + '/' + thisPlotName)
        #dill.dump(ax, file(workingFolder + '/' + thisPlotName + '.dill', 'w')) ->> CARLOS - Python 2.7
        dill.dump(ax, open(workingFolder + '/' + thisPlotName + '.dill', 'wb'))  # ->> NU - Python 3.6
        
        
        #Minimise combined distance
        bugsTargetXY2 = optimiseAllocation(parkPos, bugStatus, bugTypes, bugsTargetXY, bugTargetTypes)
        bugsTargetXY = bugsTargetXY2

        #Check for bugs outside PR
        checkValidTargetsPR(parkPos, bugsTargetXY, bugStatus, bugRouting)
        
        #Cancel motion for problem bugs
        filter = ((np.isnan(bugsTargetXY)[:,0]) & (bugStatus!=bugStatusDict['NOT_PRESENT'])) #--> NU- COMMENTED FOR TESTING - NOT_PRESENT BUGS INCLUDED NOW
        bugsTargetXY[filter]=parkPos[filter]

        
        
        
        #Start the tick array with initial and final positions
        tickArray = initialiseTickArray(parkPos, bugsTargetXY)
        initialTickArray = tickArray.copy()
        
        
        #plot initial paths
        thisPlotName = 'Initial'
        ax = plotter.plot_this(tickArray, bugStatus, [], booShowBugs = True)
        plt.plot(0,0,'b*', label='Targets')
        plt.plot(0,0,'k', label='Path', alpha=0.5, linewidth=1)
        plt.title('Direct path allocations')
        plt.legend(loc=0)
        plt.savefig(workingFolder + '/' + thisPlotName)
        #dill.dump(ax, file(workingFolder + '/' + thisPlotName + '.dill', 'w')) ->> CARLOS - Python 2.7
        dill.dump(ax, open(workingFolder + '/' + thisPlotName + '.dill', 'wb'))  # ->> NU - Python 3.6



        
        #Create crossing group array
        tempCGroups = findCrossingGroups(tickArray, bugStatus)    
        CGroups = consolidateCGroups(tempCGroups, tickArray)



        #Plot paths with groups
        thisPlotName = 'Groups'
        plotter.plot_this(tickArray, bugStatus, CGroups, booShowBugs = True)
        plt.plot(0,0,'k', label = 'Clear Direct Path', alpha = 0.7, linewidth = 7)
        plt.legend(loc=0, frameon=False, labelspacing=0.1, fontsize='small')
        plt.title('Crossing Groups Identification')
        plt.savefig(workingFolder + '/' + thisPlotName)
        #dill.dump(ax, file(workingFolder + '/' + thisPlotName + '.dill', 'w')) ->> CARLOS - Python 2.7
        dill.dump(ax, open(workingFolder + '/' + thisPlotName + '.dill', 'wb')) # ->> NU - Python 3.6




        #Solve the CGroups                                                      # NU - MAKE MORE EFFIIENT
        for i in np.unique(CGroups):
            print ('CGroupID',int(i),':', np.sum(CGroups==i), 'members.')

            CGroup = np.where(CGroups==i)[0]
            
            if i==0: #CGroup=0 is the non-crossing bugs 
                bugRouting[CGroup] = bugRoutingDict['DIRECT']
            
            else:                     
                booCollisions = True
                
                if np.sum(CGroups==i)<8: # hack to avoid endless permutations  # NU - DEF WOULD NEED TO CHANGE THIS!!!

                    #Instantiate the solver and find the moving sequence
                    thisSolver = CGroupSolver(CGroup, tickArray, bugStatus)
                    booCollisions, thisSeq, tickArray = thisSolver.findMovingSequenceBF()

                    if booCollisions==True:
                        
                        print ('Not routable. Members:', CGroup )
                        print 
                        bugRouting[CGroup] = bugRoutingDict['UNROUTABLE']
                        
                        #Copy intial position to all ticks (not moving)
                        tickArray[CGroup,1:,:] = np.reshape(np.repeat(tickArray[CGroup,0,:],tickArray.shape[1]-1, axis=0),(len(CGroup),tickArray.shape[1]-1,2))
                    else:
                        bugRouting[CGroup] = bugRoutingDict['SEQUENCE']

        #Plot ticks    
        thisTick = 0
        for thisTick in range(tickArray.shape[1]):
            thisPlotName = 'Tick'+str(thisTick)
            plotter.plot_this(tickArray[:,thisTick:thisTick+2,:], bugStatus, CGroups, booShowBugs = True)
            plt.title(thisPlotName)
            plt.savefig(workingFolder + '/' + thisPlotName)
            #dill.dump(ax, file(workingFolder + '/' + thisPlotName + '.dill', 'w')) ->> CARLOS - Python 2.7
            dill.dump(ax, open(workingFolder + '/' + thisPlotName + '.dill', 'wb'))  # ->> NU - Python 3.6


 
        #Plot paths with groups
        thisPlotName = 'SolvedGroups'
        plotter.plot_this(tickArray, bugStatus, CGroups, booShowBugs = True,  booAnnotate = True )
        plt.plot(0,0,'k', label = 'Clear Direct Path', alpha = 0.7, linewidth = 7)
        plt.legend(loc=0, frameon=False, labelspacing=0.1, fontsize='small')
        plt.title('Crossing Groups Identification')
        plt.legend(loc=0, frameon=False, labelspacing=0.1, fontsize='small')
        plt.title('Solved Crossing Groups Identification')
        plt.savefig(workingFolder + '/' + thisPlotName)
        #dill.dump(ax, file(workingFolder + '/' + thisPlotName + '.dill', 'w')) ->> CARLOS - Python 2.7
        dill.dump(ax, open(workingFolder + '/' + thisPlotName + '.dill', 'wb')) # ->> NU - Python 3.6




        '''
        Finale
        '''
        if checkForCollisions(tickArray, bugStatus):
            logger.warning('Routes created with collisions.')
        else:         
            logger.info('Routes created successfully.')

        writeOuputFile(S2FileName, S3FileName, tickArray)
        strMsg = 'File ' + S3FileName + ' written.'
        print (strMsg)
        logger.info(strMsg)

#     print 'bugRouting',bugRouting
        print ('DIRECT:', np.sum(bugRouting ==bugRoutingDict['DIRECT']) )
        print ('SEQUENCE:', np.sum(bugRouting ==bugRoutingDict['SEQUENCE']))
        print ('UNROUTABLE:', np.sum(bugRouting ==bugRoutingDict['UNROUTABLE']) )
        print ('NO_TARGET:', np.sum(bugRouting ==bugRoutingDict['NO_TARGET']) )
        print ('OUTSIDE_PR:', np.sum(bugRouting ==bugRoutingDict['OUTSIDE_PR']) )
        print ('OUTSIDE_GFP:', np.sum(bugRouting ==bugRoutingDict['OUTSIDE_GFP']) )
        print ('CRASHING:', np.sum(bugRouting ==bugRoutingDict['CRASHING']) )
        print ( 'NONE:', np.sum(bugRouting ==bugRoutingDict['NONE']) )
        
        logger.info('End of Router')
        logger.info('')
        
        return bugRouting, bugRoutingDict

    else:
        strMsg = 'File ' + S2FileName + ' not found.'
        print (strMsg)
        logger.error(strMsg)
        
    

## --------------------------------------NU - NEW---------------------------------------- ##

#def makeVidTicks(img_folder, vid_name):


## --------------------------------------NU - END ---------------------------------------- ##


#####################################################################################        
#Code starts here
#####################################################################################


if __name__ == '__main__':

    #arg parsing for command line version 
    parser = argparse.ArgumentParser(description='TaipanPyRouter. Produces a sequence of steps to move bugs from their park position to a target specified in a json (s2) file. It ouputs a Routed Tile (S3).')
    parser.add_argument('-v', help='Verbosity level (0-None, 5-Max).')  
    parser.add_argument('-o', help='Routed tile output file name (S3)', type=str, metavar='RTileFileNameS3.json')
    parser.add_argument('-f', help='Allocation target file (S2)', type=str, metavar='XYTileFileNameS2.json', default='s2_example.json')
#     parser.add_argument('-fcsv', help='filename of the output csv file', default='out.csv')
#     parser.add_argument('-fjson', help='filename of the output json file', default='out.json')
    args = parser.parse_args()
    print(args)

    doRoutes(args)
    
    





