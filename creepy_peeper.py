import numpy as np


# genetic
paths = [f"genetic/score_array{n}.npy" for n in range(15)]

print("Vi er begyndt gutter! ÙwÚ")

gen2epi = 720

final_array = np.zeros((30, 200*gen2epi, 15, 2))

#(agent, generation, track, retning)
for i, path in enumerate(paths):
    score_array = np.load(path)
    
    for n in range(2):
        #print(np.sum(final_array[i*2 + n]))
        for e in range(200):
            final_array[i*2 + n, e*gen2epi] = score_array[n, e]
        
        
    #print(path[19:21] + ":", np.shape(score_array))

for i in range(30):
    print(f"{i}: ", np.sum(final_array[i]))

print(np.shape(final_array))

np.save("final_score_genetic.npy", final_array)



"""
# vanilla to temp
paths = [f"vanilla/tester-{n1}{n2}/score_array" for n1 in range(3) for n2 in range(10)]

#(name, agent, episode, track, retning)

my_array = np.zeros((4, 8, 109000, 15, 2))
print("Vi er begyndt gutter! ÙwÚ")
for path in paths:
    
    score_array = np.load(f"{path}.npy")
    
    print(path[7:9], np.sum(my_array*score_array), flush=True)
    
    my_array += score_array
    
    #print(path, flush=True)
    
#np.save("final_score.npy", my_array)


final_array = np.load("final_score.npy")

my_dict = {}
for (n, a, e, t, b), value in np.ndenumerate(final_array):
    
    if value != 0:
        my_dict[(n, a, e, t, b)] = value
    
"""
"""
# temp tp vanilla final

vanilla_wrong = np.load("vanilla_wrong.npy")
final_score_vanilla = np.zeros((32, 109000, 15, 2))
for name in range(4):
    for agent in range(8):
        final_score_vanilla[name*8 + agent] = vanilla_wrong[name, agent]

for agent in range(32):
    print(agent, np.sum(final_score_vanilla[agent]))

np.save("final_score_vanilla.npy", final_score_vanilla)
"""