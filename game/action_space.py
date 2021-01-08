action_to_id = {
    (False, False, False, False): 0,
    (False, False, False,  True): 0,#7,
    (False, False,  True, False): 0,#8,
    (False,  True, False, False): 4,
    (False,  True, False,  True): 5,
    (False,  True,  True, False): 6,
    ( True, False, False, False): 1,
    ( True, False, False,  True): 2,
    ( True, False,  True, False): 3,
    
    
    (False, False,  True,  True): 0,
    ( True,  True, False, False): 0,
    
    ( True, False,  True,  True): 1,
    (False,  True,  True,  True): 4,
    
    ( True, True,   True,  False): 2,
    ( True, True,   False,  True): 1,
}

id_to_action = [
    (False, False, False, False),
    ( True, False, False, False),
    ( True, False, False,  True),
    ( True, False,  True, False),
    (False,  True, False, False),
    (False,  True, False,  True),
    (False,  True,  True, False),

    (False, False, False,  True),
    (False, False,  True, False),
]