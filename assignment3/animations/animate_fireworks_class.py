# Import useful libraries

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
#

class AnimateFireworks:
    def __init__(self,positions,xlim,ylim,colors,labels,filename):
        
        self.positions = positions
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim((xlim[0],xlim[1]))
        self.ax.set_ylim((ylim[0],ylim[1]))

        self.color1 = colors[0]
        self.color2 = colors[1] 

        self.label1 = labels[0]
        self.label2 = labels[1] 

        self.filename = filename

        #self.ani = animation.FuncAnimation(fig=self.fig, func=self.update, frames=np.arange(0,len(positions)-1,1000),fargs=(self,))

    def update(frame, args):
        
        positions = args[0]
        color1 = args[1]
        color2 = args[2]
        label1 = args[3]
        label2 = args[4]
        ax = args[5]
        
        x1 = positions[:frame, 0, 0]
        x2 = positions[:frame, 1, 0]
        y1 = positions[:frame, 0, 1]
        y2 = positions[:frame, 1, 1]

        # update the scatter plot:
        ax.scatter(x1, y1, c=color1, s=5, label=label1)
        ax.scatter(x2, y2, c=color2, s=5, label=label2)
        


    def save(self):

        args = {"pos": self.positions,
                "colors":[self.color1,self.color2],
                "labels":[self.label1,self.label2],
                "ax": self.ax}
        
        ani = animation.FuncAnimation(fig=self.fig, 
                                      func=self.update, 
                                      frames=np.arange(0,len(self.positions)-1,1000),
                                      fargs=(args),
                                      )
        ani.save(self.filename)
        print("ho salvato")

    



"""
if __name__ == "__main__":
    particles = ...  # Define the particles object
    xlim = ...  # Define the x-axis limits
    ylim = ...  # Define the y-axis limits
    colors = ...  # Define the colors for the particles
    labels = ...  # Define the labels for the particles
    filename = ...  # Define the filename for saving the animation

    fireworks_animation = animate_fireworks(particles, xlim, ylim, colors, labels, filename)
    fireworks_animation.show()
"""