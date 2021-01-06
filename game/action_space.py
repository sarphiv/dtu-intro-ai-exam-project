action_to_id = {
    (False, False, False, False): 0,
    (False, False, False,  True): 1,
    (False, False,  True, False): 2,
    (False,  True, False, False): 3,
    (False,  True, False,  True): 4,
    (False,  True,  True, False): 5,
    ( True, False, False, False): 6,
    ( True, False, False,  True): 7,
    ( True, False,  True, False): 8,
    
    
    (False, False,  True,  True): 0,
    ( True,  True, False, False): 0,
    
    ( True, False,  True,  True): 6,
    (False,  True,  True,  True): 3,
    
    ( True, True,   True,  False): 2,
    ( True, True,   False,  True): 1,
}

id_to_action = [
    (False, False, False, False),
    (False, False, False,  True),
    (False, False,  True, False),
    (False,  True, False, False),
    (False,  True, False,  True),
    (False,  True,  True, False),
    ( True, False, False, False),
    ( True, False, False,  True),
    ( True, False,  True, False),
]