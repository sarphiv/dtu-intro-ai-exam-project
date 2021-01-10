action_to_id = {
    (False, False, False, False): 0,
    (False, False, False,  True): 0,#7,
    (False, False,  True, False): 0,#8,
    (False,  True, False, False): 4,
    (False,  True, False,  True): 6,#Should be 5 for human controls
    (False,  True,  True, False): 5,#Should be 6 for human controls
    ( True, False, False, False): 1,
    ( True, False, False,  True): 3,
    ( True, False,  True, False): 2,
    
    
    (False, False,  True,  True): 0,
    ( True,  True, False, False): 0,
    
    ( True, False,  True,  True): 1,
    (False,  True,  True,  True): 4,
    
    ( True, True,   True,  False): 0,#7,
    ( True, True,   False,  True): 0,#8,
}

id_to_action = [
    (False, False, False, False),
    ( True, False, False, False),
    ( True, False,  True, False),
    ( True, False, False,  True),
    (False,  True, False, False),
    (False,  True,  True, False),
    (False,  True, False,  True),

    (False, False,  True, False),
    (False, False, False,  True),
]