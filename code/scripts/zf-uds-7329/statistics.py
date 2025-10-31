def calculate_chisq(data, model, error):
    """Simple function to calculate the chi-squared value of a model, given the data and associated error"""

    chisq = (data - model)**2 / error**2

    return chisq

def calculate_reduced_chisq(chisq, ndof):
    """Simple function to calculate the reduced chi-squared value of a model, given the chisquared value and degrees of freedom of the model"""

    red_chisq = chisq / ndof

    return red_chisq