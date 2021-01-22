import numpy as np


# (agent, episode, track, retning)

interresting_score_limit = 575000
model_types = ["final_score_vanilla.npy", "final_score_genetic.npy"]

def get_value(score_array):
    """
    # bedste inden episode
    reduced = score_array[:5500]
    argmax = reduced.argmax()
    return(argmax, reduced[argmax])

    # sidste
    last = np.where(score_array != 0)[0][-1]
    return(last, score_array[last])
    
    # bedste
    argmax = score_array.argmax()
    return(argmax, score_array[argmax])
    """
    # score over
    if max(score_array) > interresting_score_limit:
        first = np.where(score_array > interresting_score_limit)[0][0]
        score = score_array[first]
    else:
        first = len(score_array)
        score = 0
        
    return(first, score)

tofile = []
for model_i, model_type in enumerate(model_types):
    final_array = np.load(model_type)
    for agent_i, agent in enumerate(final_array):
        score_array = agent.mean(axis=(-2,-1))
        episode, score = get_value(score_array)
        print(model_i, agent_i)
        tofile.append([episode, score, model_i])


#print(tofile)
np.savetxt("temp_data.csv", tofile, delimiter=',', header=r"episode, score, model")


"""
from plotting.CsvManager import CsvManager
from plotting.PlotManager import PlotManager

csvM = CsvManager(["Games Played", "Average reward"], file_name="plot-data/c00.csv")
plotM = PlotManager([csvM])

final_array = np.load(model_types[0])
scores = final_array[18].mean(axis=(1,2))
for episode, score in enumerate(scores):
    if score != 0:
        #print(episode, score)
        csvM.save_data([episode, score])

plotM.plot()
"""