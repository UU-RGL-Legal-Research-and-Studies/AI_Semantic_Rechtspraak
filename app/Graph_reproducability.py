# Update the plot with the final requested change to remove 'over meerdere runs' from the title
plt.figure(figsize=(10, 6))

# Plot the scores of each run
for ecli in df['ECLI'].unique():
    subset = df[df['ECLI'] == ecli]
    plt.plot(subset['Run'], subset['Score'], label=ecli, marker='o', linestyle='-')

plt.title('Reproduceerbaarheid van semantische vergelijkbaarheidsscores')
plt.xlabel('Runnummer')
plt.ylabel('Semantische vergelijkbaarheidsscore')
plt.legend(title='ECLI', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the final plot
plt.tight_layout()
plt.show()
