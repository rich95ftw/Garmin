import fitdecode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Helper function to format elapsed time as MM:SS
def format_mmss(x, pos):
    total_seconds = int(x * 60)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"

# ------------------------
# 1. Parse FIT file
# ------------------------
records = []
with fitdecode.FitReader("19179254886_ACTIVITY.fit") as fit:
    for frame in fit:
        if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
            record = {field.name: field.value for field in frame.fields}
            records.append(record)

# ------------------------
# 2. Create DataFrame
# ------------------------
df = pd.DataFrame(records)

# Keep only relevant columns and drop rows with missing data
df = df[['timestamp', 'heart_rate', 'enhanced_speed']].dropna()
df.rename(columns={'enhanced_speed': 'speed'}, inplace=True)

# ------------------------
# 3. Add elapsed time and pace
# ------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])  # ensure timestamp is datetime
df['elapsed_time'] = df['timestamp'] - df['timestamp'].iloc[0]
df['elapsed_minutes'] = df['elapsed_time'].dt.total_seconds() / 60
df['pace_min_per_km'] = 1000 / (df['speed'] * 60)  # pace = 1000 m / (speed × 60)

# ------------------------
# 4. Plot Heart Rate vs Elapsed Time
# ------------------------
plt.figure(figsize=(12, 4))
plt.plot(df['elapsed_minutes'], df['heart_rate'], color='red')
plt.xlabel('Elapsed Time')
plt.ylabel('Heart Rate (bpm)')
plt.title('Heart Rate vs Elapsed Time')
plt.grid(True)

# Format x-axis as mm:ss
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_mmss))

plt.tight_layout()
plt.show()
# ------------------------
# 5. Plot Pace vs Elapsed Time
# ------------------------
plt.figure(figsize=(12, 4))
plt.plot(df['elapsed_minutes'], df['pace_min_per_km'], color='blue')
plt.xlabel('Elapsed Time')
plt.ylabel('Pace (min/km)')
plt.title('Pace vs Elapsed Time')
plt.gca().invert_yaxis()
plt.grid(True)

# Format x-axis as mm:ss
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_mmss))

plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 5))

# Heart Rate on left y-axis
ax1.plot(df['elapsed_minutes'], df['heart_rate'], color='red', label='Heart Rate (bpm)')
ax1.set_xlabel('Elapsed Time (mm:ss)')
ax1.set_ylabel('Heart Rate (bpm)', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True)

ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_mmss))

# Pace on right y-axis
ax2 = ax1.twinx()
ax2.plot(df['elapsed_minutes'], df['pace_min_per_km'], color='blue', label='Pace (min/km)')
ax2.set_ylabel('Pace (min/km)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.invert_yaxis()

plt.title('Heart Rate and Pace vs Elapsed Time')
fig.tight_layout()

# Legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.show()