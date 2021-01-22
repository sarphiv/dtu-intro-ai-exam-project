from csv_manager import csv_manager 
from plot_manager import plot_manager

# for i in range(30):
#     csv_random_reward = csv_manager(["Games Played", "Avg Reward"], file_name=f"data/outside_{i+1}.csv", clear=False)
#     plot_random_reward = plot_manager([csv_random_reward])
#     plot_random_reward.plot()

# for i in range(10):
#     csv_random_reward = csv_manager(["Games Played", "Avg Reward"], file_name=f"C:\\Users\david\Desktop\Toren driller\genetic-lunar{i+1}\plot-data/current-plot.csv", clear=False)
#     plot_random_reward = plot_manager([csv_random_reward])
#     plot_random_reward.plot()

csv1 = csv_manager(["Games Played", "Avg Reward"], file_name=f"data/outside_{1}.csv", clear=False)
csv2 = csv_manager(["Games Played", "Avg Reward"], file_name=f"data/outside_{7}.csv", clear=False)
csv3 = csv_manager(["Games Played", "Avg Reward"], file_name=f"data/outside_{8}.csv", clear=False)
plot_random_reward = plot_manager([csv1, csv2, csv3])
plot_random_reward.plot()

input("torben er dum og driller : ")





