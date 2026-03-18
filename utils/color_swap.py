import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Trend Verdict Colors
content = content.replace('verdict["trend"] = "⚠️ 上升楔形"; verdict["color"] = "red"', 'verdict["trend"] = "⚠️ 上升楔形"; verdict["color"] = "green"')
content = content.replace('verdict["trend"] = "🟢 多頭趨勢"; verdict["color"] = "green"', 'verdict["trend"] = "🔴 多頭趨勢"; verdict["color"] = "red"')
content = content.replace('verdict["trend"] = "✨ 下降楔形"; verdict["color"] = "green"', 'verdict["trend"] = "✨ 下降楔形"; verdict["color"] = "red"')
content = content.replace('verdict["trend"] = "🔴 空頭趨勢"; verdict["color"] = "red"', 'verdict["trend"] = "🟢 空頭趨勢"; verdict["color"] = "green"')

# 2. Candlestick Core Plotly
content = content.replace("increasing_line_color='#ef4444', decreasing_line_color='#22c55e'", "increasing_line_color='#ef4444', decreasing_line_color='#22c55e'") # Wait, original was red/green, so wait... #ef4444 is Red. #22c55e is Green. Wait, if it was already Red/Green? Let's check original!
