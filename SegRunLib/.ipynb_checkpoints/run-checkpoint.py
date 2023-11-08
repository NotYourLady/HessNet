from SegRunLib.scripts.RunnerClass import Runner

def run(settings=None):
    runner = Runner(settings=settings)
    runner.predict_and_save(settings['in_path_nifty'], settings['out_path_nifty'])