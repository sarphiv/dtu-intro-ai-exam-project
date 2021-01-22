
# # %%
# from plotting.CsvManager import CsvManager
# from plotting.PlotManager import PlotManager

# gen7 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{7}", clear=False)
# gen15 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{15}", clear=False)
# gen3 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{3}", clear=False)
# gen28 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{28}", clear=False)

# van1 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project/vanilla_plots/random_reward_{1}.csv", clear=False)
# van12 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project/vanilla_plots/random_reward_{12}.csv", clear=False)
# van14 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project/vanilla_plots/random_reward_{14}.csv", clear=False)
# van21 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project/vanilla_plots/random_reward_{21}.csv", clear=False)

# plot_random_reward = PlotManager([gen3, gen7, gen15, gen28, van1], label=False)
# #plot_random_reward = PlotManager([gen3, gen7, gen15, gen22, van1, van12, van14, van21], label=False)
# #plot_random_reward = PlotManager([van1, van12, van14, van21], label=False)
# plot_random_reward.plot()

# input("torben er dum og driller : ")


#%%
from plotting.CsvManager import CsvManager
from plotting.PlotManager import PlotManager

# for i in range(30):
#     csv_random_reward = csv_manager(["Games Played", "Avg Reward"], file_name=f"data/random_reward_{i+1}.csv", clear=False)
#     plot_random_reward = plot_manager([csv_random_reward])
#     plot_random_reward.plot()

# for i in range(1, 31):
#     csv_random_reward = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{i}", clear=False)
#     plot_random_reward = PlotManager([csv_random_reward])
#     plot_random_reward.plot()

# #gen5 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{5}", clear=False)
# gen7 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{7}", clear=False)
# gen15 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{15}", clear=False)
# #gen18 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{18}", clear=False)
# gen3 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{3}", clear=False)
# gen29 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{16}", clear=False)


# van1 = CsvManager(["Games Played", "Avg Reward"], file_name="C:\\Users\david\Desktop\LunarLander - first Main - Copy\dtu-intro-ai-exam-project\data\random_reward_1.csv", clear=False)
# van12 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\LunarLander - first Main - Copy\dtu-intro-ai-exam-project\data\random_reward_12.csv", clear=False)
# van14 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\LunarLander - first Main - Copy\dtu-intro-ai-exam-project\data\random_reward_14.csv", clear=False)
# van21 = CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\LunarLander - first Main - Copy\dtu-intro-ai-exam-project\data\random_reward_21.csv", clear=False)

# plot_random_reward = PlotManager([gen3, gen7, gen15, gen29], label=False) #, van1, van12, van14, van21])
# plot_random_reward.plot()

#%%
from plotting.CsvManager import CsvManager
from plotting.PlotManager import PlotManager
managers = []
for i in range(1, 31): 
    managers.append(CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project\plot-data/plot_file_reward{i}", clear=False))

plot_random_reward = PlotManager(managers, label=False) #, van1, van12, van14, van21])
plot_random_reward.plot()
input("torben er dum og driller : ")

#%%
from plotting.CsvManager import CsvManager
from plotting.PlotManager import PlotManager
managers = []
for i in range(1, 31): 
    managers.append(CsvManager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\gentic-evaluation\dtu-intro-ai-exam-project/vanilla_plots/random_reward_{i}.csv", clear=False))

plot_random_reward = PlotManager(managers, label=False) #, van1, van12, van14, van21])
plot_random_reward.plot()
input("torben er dum og driller : ")



# %%
