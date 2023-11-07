from SegRunLib.scripts.RunnerClass import Runner

def run(in_path_nifty: str, out_path_nifty: str, settings=None):
    runner = Runner(settings=settings)
    runner.predict_and_save(in_path_nifty, out_path_nifty)
    
    
    