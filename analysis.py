import fitdecode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

# Apply exponential moving average smoothing with a span parameter
df['heart_rate_smooth'] = df['heart_rate'].ewm(span=10, adjust=False).mean()

# Plot raw and smoothed heart rate
plt.figure(figsize=(12, 4))
plt.plot(df['elapsed_minutes'], df['heart_rate'], color='lightcoral', alpha=0.5, label='Raw Heart Rate')
plt.plot(df['elapsed_minutes'], df['heart_rate_smooth'], color='red', label='Smoothed Heart Rate (EMA)')
plt.xlabel('Elapsed Time (mm:ss)')
plt.ylabel('Heart Rate (bpm)')
plt.title('Heart Rate with Exponential Smoothing')
plt.grid(True)
plt.legend()

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_mmss))
plt.tight_layout()
plt.show()

# Smooth pace data using EMA with the same span (you can adjust span if needed)
df['pace_smooth'] = df['pace_min_per_km'].ewm(span=10, adjust=False).mean()

fig, ax1 = plt.subplots(figsize=(12, 5))

# Plot smoothed Heart Rate on left y-axis
ax1.plot(df['elapsed_minutes'], df['heart_rate_smooth'], color='red', label='Smoothed Heart Rate (EMA)')
ax1.set_xlabel('Elapsed Time (mm:ss)')
ax1.set_ylabel('Heart Rate (bpm)', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True)

ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_mmss))

# Plot smoothed Pace on right y-axis
ax2 = ax1.twinx()
ax2.plot(df['elapsed_minutes'], df['pace_smooth'], color='blue', label='Smoothed Pace (min/km)')
ax2.set_ylabel('Pace (min/km)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.invert_yaxis()  # Faster pace is lower on the graph

plt.title('Smoothed Heart Rate and Pace vs Elapsed Time')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.tight_layout()
plt.show()

# Calculate heart rate as % of max heart rate (180 bpm)
df['heart_rate_pct_max'] = (df['heart_rate_smooth'] / 180) * 100

fig, ax = plt.subplots(figsize=(12, 5))

# Plot heart rate % max
ax.plot(df['elapsed_minutes'], df['heart_rate_pct_max'], color='red', label='Heart Rate (% of Max)')

# Define zones boundaries again (in % max HR)
zones = {
    "Zone 5 (>159 bpm)": (88.3, 110),
    "Zone 4 (141-159 bpm)": (78.3, 88.3),
    "Zone 3 (125-140 bpm)": (69.4, 77.8),
    "Zone 2 (107-124 bpm)": (59.4, 68.9),
    "Zone 1 (<107 bpm)": (0, 59.4),
}

zone_colors = {
    "Zone 5 (>159 bpm)": '#ff6666',
    "Zone 4 (141-159 bpm)": '#ffcc66',
    "Zone 3 (125-140 bpm)": '#ffff99',
    "Zone 2 (107-124 bpm)": '#99ccff',
    "Zone 1 (<107 bpm)": '#cce5ff',
}

# Assign each row to a zone
def assign_zone(hr_pct):
    for zone, (low, high) in zones.items():
        if low <= hr_pct < high:
            return zone
    return "Unknown"

df['hr_zone'] = df['heart_rate_pct_max'].apply(assign_zone)

# Calculate time spent in each zone
# Since elapsed_minutes is cumulative, get the diff for intervals (in minutes)
df['elapsed_diff'] = df['elapsed_minutes'].diff().fillna(0)

# Sum the time intervals per zone
time_in_zones = df.groupby('hr_zone')['elapsed_diff'].sum()

# Convert times to mm:ss format for display
def format_time(mins):
    total_seconds = int(mins * 60)
    return f"{total_seconds // 60}:{total_seconds % 60:02}"

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df['elapsed_minutes'], df['heart_rate_pct_max'], color='red', label='Heart Rate (% of Max)')

for zone, (low, high) in zones.items():
    ax.axhspan(low, high, color=zone_colors[zone], alpha=0.3, label=zone)

# Add time labels inside zones (left side)
for i, (zone, (low, high)) in enumerate(zones.items()):
    time_spent = time_in_zones.get(zone, 0)
    time_str = format_time(time_spent)
    # Place label at middle height of zone and near left (e.g., at 1 min elapsed)
    ax.text(1, (low + high) / 2, f"{zone}\n{time_str} min", 
            va='center', ha='left', fontsize=9, fontweight='bold', color='black', alpha=0.7)

ax.set_xlabel('Elapsed Time (mm:ss)')
ax.set_ylabel('Heart Rate (% of Max)')
ax.set_ylim(0, 100)
ax.grid(True)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_mmss))

plt.title('Smoothed Heart Rate as % of Max with Training Zones & Time Spent')
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.show()