import fitdecode
import pandas as pd
import matplotlib.pyplot as plt

records = []

with fitdecode.FitReader("19179254886_ACTIVITY.fit") as fit:
    for frame in fit:
        if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
            record = {field.name: field.value for field in frame.fields}
            records.append(record)

df = pd.DataFrame(records)

# Filter rows with valid time and heart_rate/speed
df = df[['timestamp', 'heart_rate', 'speed']].dropna()

df['pace_min_per_km'] = 1000 / (df['speed'] * 60)  # pace = 1000 meters / (speed * 60 seconds)

plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], df['heart_rate'], color='red')
plt.title("Heart Rate vs Time")
plt.xlabel("Time")
plt.ylabel("Heart Rate (bpm)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], df['pace_min_per_km'], color='blue')
plt.title("Pace vs Time")
plt.xlabel("Time")
plt.ylabel("Pace (min/km)")
plt.gca().invert_yaxis()  # Faster pace is "lower" on the graph
plt.grid(True)
plt.tight_layout()
plt.show()
