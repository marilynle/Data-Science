"""
Wrangle function to create boolean columns for most important amenities
"""
import pandas as pd

def wrangle(df):
    """
    Creates boolean columns
    for most frequent amenities
    """

    used_amenities = ['Washer', 'Hair dryer', 'Laptop friendly workspace',
                          'Hangers', 'Iron', 'Shampoo', 'TV', 'Hot water',
                          'Family/kid friendly', 'Internet', 'Host greets you',
                          'Smoke detector', 'Buzzer/wireless intercom',
                          'Lock on bedroom door', 'Free street parking', 'Elevator',
                          'Bed linens', 'Smoking allowed', 'First aid kit', 'Cable TV']

    # Create Boolean columns for each amenity
    for item in used_amenities:
        df[item] = df['amenities'].str.contains(item).astype(int)

    # Drop the original amenities column
    df = df.drop(columns='amenities')

    return df
