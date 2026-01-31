def dim_mulitplier(selector_string):
    if selector_string == 'long':
        return (1, 3)
    elif selector_string == 'wide':
        return (3, 1)
    elif selector_string == 'square':
        return (2,2)
    else:
        raise ValueError(f'{slector_string} is not supported: options are long, square, or wide')