import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('stats.csv')

# a) Distribution of mean temperatures (histogram)
plt.figure(figsize=(10, 6))
plt.hist(data['mean_temp'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Mean Temperatures Across Buildings')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Number of Buildings')
plt.grid(True, alpha=0.3)
plt.savefig('temperature_distribution.png')
plt.close()

# b) Average mean temperature across buildings
avg_mean_temp = data['mean_temp'].mean()
print(f"b) Average mean temperature across buildings: {avg_mean_temp:.2f}°C")

# c) Average temperature standard deviation
avg_std_temp = data['std_temp'].mean()
print(f"c) Average temperature standard deviation: {avg_std_temp:.2f}°C")

# d) Number of buildings with at least 50% area above 18°C
buildings_above_18 = (data['pct_above_18'] >= 50).sum()
print(f"d) Buildings with at least 50% area above 18°C: {buildings_above_18} ({buildings_above_18/len(data)*100:.1f}%)")

# e) Number of buildings with at least 50% area below 15°C
buildings_below_15 = (data['pct_below_15'] >= 50).sum()
print(f"e) Buildings with at least 50% area below 15°C: {buildings_below_15} ({buildings_below_15/len(data)*100:.1f}%)")

# Additional insights
print("\nAdditional Statistics:")
print(f"Total number of buildings analyzed: {len(data)}")
print(f"Mean temperature range: {data['mean_temp'].min():.2f}°C to {data['mean_temp'].max():.2f}°C")
print(f"Average percentage of area above 18°C: {data['pct_above_18'].mean():.2f}%")
print(f"Average percentage of area below 15°C: {data['pct_below_15'].mean():.2f}%")

# Create a histogram for percentage above 18°C
plt.figure(figsize=(10, 6))
plt.hist(data['pct_above_18'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Percentage Area Above 18°C')
plt.xlabel('Percentage of Area Above 18°C')
plt.ylabel('Number of Buildings')
plt.grid(True, alpha=0.3)
plt.savefig('above_18_distribution.png')
plt.close()

# Create a histogram for percentage below 15°C
plt.figure(figsize=(10, 6))
plt.hist(data['pct_below_15'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Percentage Area Below 15°C')
plt.xlabel('Percentage of Area Below 15°C')
plt.ylabel('Number of Buildings')
plt.grid(True, alpha=0.3)
plt.savefig('below_15_distribution.png')
plt.close()