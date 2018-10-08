import pylab as plt
import numpy as np
import constants as const

def draw_canvas():
    """
    Draw an empty GFP canvas
    
    Args:
        
    Returns:
        ax,fig of the created canvas 
        
    Note:
    """
   
    
    #clear plot
    plt.clf()
    fig = plt.gcf().gca()
    ax = plt.subplot()
    
    #GFP
    circle1 = plt.Circle((0, 0), const.GFP_RADIUS, color='k', alpha = 0.3)
    ax.add_artist(circle1)
    
    #patrol radius
#     circle1 = plt.Circle((0, 0), const.PATROL_RADIUS, color='k', alpha = 0.3)
#     ax.add_artist(circle1)

    plt.ylim((-180000, 180000))
    plt.xlim((-180000, 180000))
    plt.xlabel('X position (microns)')
    plt.ylabel('Y position (microns)')
    plt.legend(loc=0)
    ax.set_aspect('equal')
    return ax,fig


#Plots a complete path for a set of bugs
def plot_this(tickArray, bugStatus, CGroups, booLegend = False, booShowBugs = False, booAnnotate = False, booThickPath = True):
    
    plt.clf()
    ax,fig = draw_canvas()
    
    if CGroups!=[]:    
        colorArray = plt.cm.jet(np.linspace(0,1,np.max(CGroups)+1))
        lwidth = 5
    else:
        colorArray = plt.cm.rainbow(np.linspace(0,1,100))
        lwidth = 1
        
    lineStyleArray = ['-', '--', '-.', ':']
    lineStyleArray = ['-', '-', '-', '-']
    
    
#     for thisBugIdx,thisBug in zip(np.arange(tickArray.shape[0])[bugStatus==0],tickArray[bugStatus==0]):
    for thisBugIdx,thisBug in zip(np.arange(tickArray.shape[0]),tickArray):

        if np.isnan(np.sum(thisBug))==False:
            
            thisBugXs, thisBugYs =  thisBug.transpose()
#             if not ((thisBugXs[0]-thisBugXs[-1]==0) and (thisBugYs[0]-thisBugYs[-1]==0)):
            if 0==0:    
                if booShowBugs==True:
                    for i, thisX, thisY in zip(range(thisBugXs.shape[0]),thisBugXs,thisBugYs):
                        if thisX==thisBugXs[0]: #the first one
                            circle1 = plt.Circle((thisX,thisY), const.BUG_RADIUS, color='k', alpha = 0.3)
                            ax.add_artist(circle1)
        
                        elif i< thisBugXs.shape[0]: #the middle ones
                            plt.plot([thisX,thisX],[thisY,thisY], '*b', markersize=5)
                            circle1 = plt.Circle((thisX,thisY), const.BUG_RADIUS, color='g', alpha = 0.5)
                            ax.add_artist(circle1)
        
                        else: #the last one
                            circle1 = plt.Circle((thisX,thisY), const.BUG_RADIUS, color='g', alpha = 0.1)
                            ax.add_artist(circle1)
        
                        if booAnnotate:
                                plt.annotate(
                                    thisBugIdx,
                                    xy=(thisX, thisY), xytext=(0, 20),
                                    textcoords='offset points', ha='right', va='bottom',
                                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        
                if CGroups!=[]:                        
                    if CGroups[thisBugIdx]==0:
                        col = 'k'
                    else:
                        col = colorArray[int(CGroups[thisBugIdx])]
                    
                    lin = lineStyleArray[int(CGroups[thisBugIdx]%4)]
                else:
                    col = 'k'
                    lin = lineStyleArray[0]
                    
                plt.plot(thisBugXs, thisBugYs,color = col,linestyle=lin, alpha = 0.5, linewidth = lwidth)
#         else:
#             print thisBugIdx,thisBug
#             print np.isnan(np.sum(thisBug)),np.sum(thisBug)
#             print 
                
    if booLegend: plt.legend(loc=0)
#     fig.canvas.draw()
    return ax

# =========================    NU ADDITION    ========================== #
#Plot routes with bug types in different colours
def plot_this_with_bugtypes(tickArray, bugStatus, CGroups, bugTypes, booLegend=False, booShowBugs=False, booAnnotate=False, booThickPath=True):
    plt.clf()
    ax, fig = draw_canvas()

    if CGroups != []:
        colorArray = plt.cm.jet(np.linspace(0, 1, np.max(CGroups) + 1))
        lwidth = 5
    else:
        colorArray = plt.cm.rainbow(np.linspace(0, 1, 100))
        lwidth = 1

    lineStyleArray = ['-', '--', '-.', ':']
    lineStyleArray = ['-', '-', '-', '-']

    #     for thisBugIdx,thisBug in zip(np.arange(tickArray.shape[0])[bugStatus==0],tickArray[bugStatus==0]):
    for thisBugIdx, thisBug in zip(np.arange(tickArray.shape[0]), tickArray):

        if np.isnan(np.sum(thisBug)) == False:

            thisBugXs, thisBugYs = thisBug.transpose()
            #             if not ((thisBugXs[0]-thisBugXs[-1]==0) and (thisBugYs[0]-thisBugYs[-1]==0)):
            if 0 == 0:
                if booShowBugs == True:
                    for i, thisX, thisY in zip(range(thisBugXs.shape[0]), thisBugXs, thisBugYs):
                        if thisX == thisBugXs[0]:  # the first one
                            circle1 = plt.Circle((thisX, thisY), const.BUG_RADIUS, color='k', alpha=0.3)
                            ax.add_artist(circle1)

                        elif i < thisBugXs.shape[0]:  # the middle ones
                            plt.plot([thisX, thisX], [thisY, thisY], '*b', markersize=5)
                            circle1 = plt.Circle((thisX, thisY), const.BUG_RADIUS, color='g', alpha=0.5)
                            ax.add_artist(circle1)

                        else:  # the last one
                            circle1 = plt.Circle((thisX, thisY), const.BUG_RADIUS, color='g', alpha=0.1)
                            ax.add_artist(circle1)

                        if booAnnotate:
                            plt.annotate(
                                thisBugIdx,
                                xy=(thisX, thisY), xytext=(0, 20),
                                textcoords='offset points', ha='right', va='bottom',
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

                if CGroups != []:
                    if CGroups[thisBugIdx] == 0:
                        col = 'k'
                    else:
                        col = colorArray[int(CGroups[thisBugIdx])]

                    lin = lineStyleArray[int(CGroups[thisBugIdx] % 4)]
                else:
                    col = 'k'
                    lin = lineStyleArray[0]

                plt.plot(thisBugXs, thisBugYs, color=col, linestyle=lin, alpha=0.5, linewidth=lwidth)
    #         else:
    #             print thisBugIdx,thisBug
    #             print np.isnan(np.sum(thisBug)),np.sum(thisBug)
    #             print

    if booLegend: plt.legend(loc=0)
    #     fig.canvas.draw()
    return ax

# =========================    NU ADDITION- END    ========================== #

#Plots all park positions
def plot_park_pos(parkPos, bugStatus, bugTypes, booLegend = False, booShowBugs = True, booAnnotate = False, booThickPath = True):
    
    plt.clf()
    ax,fig = draw_canvas()
        
    #hack for the  label
#     plt.plot(0, 0, '.k', label='Patrol Radius')
#     plt.plot(0, 0, '.b', label='Science Starbugs')
#     plt.plot(0, 0, '.g', label='Guide Starbugs')
    

    for i in range(parkPos.shape[0]):
        thisBugType = bugTypes[i]
        thisBugX = parkPos[i,0]
        thisBugY = parkPos[i,1]
        thisBugStatus = bugStatus[i]

        #Patrol radius
#         if ((thisBugX==0) and (thisBugY==0)):
#             circle1 = plt.Circle((thisBugX,thisBugY), const.PATROL_RADIUS , color='k', alpha = 0.3)
#             ax.add_artist(circle1)
#             circle1 = plt.Circle((thisBugX,thisBugY), const.PATROL_RADIUS , color='k', alpha = 1, linewidth=1, fill=False)
#             ax.add_artist(circle1)
            
        #Green for guide, blue for the rest    
        if thisBugType==0:
            c = 'g'
        else:
            c = 'b'
        
        #override if bug not working
        if ((thisBugStatus==1) or (thisBugStatus==2)): 
            c = 'r'
            
        if thisBugStatus!=3: 
            bugShape = plt.Circle((thisBugX,thisBugY), const.BUG_RADIUS , color=c, alpha = 0.5, )
            ax.add_artist(bugShape)
        

    plt.title('Starbugs Park Positions')
    
    return ax



#Plots all targets in the field
def plot_targets(bugsTargetXY, bugTargetTypes, booLegend = False, booShowBugs = True, booAnnotate = False, booThickPath = True):
    
    plt.clf()
    ax,fig = draw_canvas()
        
    for i in range(bugsTargetXY.shape[0]):
        thisTargetType = bugTargetTypes[i]
        thisTargetX = bugsTargetXY[i,0]
        thisTargetY = bugsTargetXY[i,1]

        #Green for guide, blue for the rest    
        if thisTargetType==0:
            c = 'g'
        else:
            c = 'b'

        bugShape = plt.Circle((thisTargetX,thisTargetY), const.BUG_RADIUS , color=c, alpha = 0.5, )
        ax.add_artist(bugShape)
        

    plt.title('Target Positions')
    
    return ax
