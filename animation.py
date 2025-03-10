import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress
import time,io
from PIL import Image
import imageio
from plotly.subplots import make_subplots


class MoviePlayer:
    def __init__(self, video_array, n_plots=1,interval=30):
        self.video_array = video_array
        self.num_frames  = video_array.shape[0]
        self.interval    = interval
        self.n_plots     = n_plots
        self.plot_hooks  = []
        
    def create_trace(self,t):
        hook_results = []
        futures={}
        with ThreadPoolExecutor() as executor:
            for idx,hook in enumerate(self.plot_hooks):
                futures[executor.submit(hook, t,self.video_array)]= (1,idx+1)

            for f in as_completed(futures.keys()):
                row,col = futures[f]
                result = f.result()
                if not isinstance(result,list):
                    result = [result]
                for res in result:
                    res.name =f'rc_{row}{col}'
                    hook_results.append(dict(row=row,col=col,data=res))
        return (t,hook_results)
                    
    def register(self, hooks):
        """
        Registers one or more hooks to be called on each frame update.

        Parameters:
        hooks (callable or list of callables): Functions to be called with the current axes and video array.
        """
        if not isinstance(hooks, list):
            hooks = [hooks]
        self.plot_hooks = hooks

    def plotly_fig2array(self,trace_lt):
        fig = make_subplots(rows=1,cols=self.n_plots)
        for trace in trace_lt:
            fig.add_trace(trace['data'],row=trace['row'],col=trace['col'])
        fig_bytes = fig.to_image(format="png")
        buf = io.BytesIO(fig_bytes)
        img = Image.open(buf)
        return np.asarray(img)


    def play(self):
        result = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.create_trace,t)  for t in range(self.num_frames)]
            with Progress() as progress:
                task = progress.add_task("[green]Processing...", total=len(futures))
                for idx,f in enumerate(as_completed(futures)):
                    res = f.result() 
                    result.append(res)
                    progress.update(task, advance=1)

            
        result = sorted(result,key=lambda x: x[0],reverse=False)
               
        frames = []
        st = time.time()
        with ThreadPoolExecutor(max_workers=len(result)) as executor:
            futures = [executor.submit(self.plotly_fig2array,res[1])  for res in result]
            for f in as_completed(futures):
                arr = f.result()
                frames.append(arr)
        imageio.mimsave('mitralvalve.gif', frames, fps=1)



