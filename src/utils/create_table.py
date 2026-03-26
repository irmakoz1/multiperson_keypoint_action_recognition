import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file (replace 'your_file.csv' with your actual file path)
df = pd.read_csv('results.csv')

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 4))  # adjust size as needed
ax.axis('off')  # hide the axes

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center',
                 cellLoc='center', colColours=['#f2f2f2']*len(df.columns))

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # adjust column width and row height

# Apply colors to rows (alternating colors)
for (row, col), cell in table.get_celld().items():
    if row == 0:  # header row
        cell.set_facecolor('#4CAF50')  # green header
        cell.set_text_props(weight='bold', color='white')
    else:
        if row % 2 == 0:
            cell.set_facecolor('#f9f9f9')
        else:
            cell.set_facecolor('#ffffff')

# Add border to cells
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#cccccc')
    cell.set_linewidth(0.5)

# Save as PNG (high DPI for clarity)
plt.savefig('table_for_presentation.png', dpi=300, bbox_inches='tight')
plt.show()