import re

code = """
        # Candle Patterns Annotations
        if st.session_state.chart_settings.get('candle_patterns', True) and candle_patterns:
            annotation_counts = {}
            for p in candle_patterns:
                date = p['date']

                # Check for "利多" or "利空" in the text description or name
                # Actually, the user says "所有關於利多的字，顏色都幫我換成紅色，反之則為綠色"
                # This could mean that the color of the pattern depends on if it's bullish or bearish,
                # OR it could mean literally if the text says "利多", color it red.
                # In traditional finance, Bullish = 利多 (Red), Bearish = 利空 (Green).

                is_bearish = p['type'] == 'Bearish'
                y_val = df.loc[date, 'High'] * 1.02 if is_bearish else df.loc[date, 'Low'] * 0.98

                # Font color matches TW standard (Red for Bullish, Green for Bearish)
                font_color = "green" if is_bearish else "red"
                clean_name = p['name']

                # Handle overlap counting
                key = (date, is_bearish)
                count = annotation_counts.get(key, 0)
                annotation_counts[key] = count + 1

                if len(p.get('points', [])) > 1:
                    # Continuous pattern: use large quotation marks
                    offset_y = (count * 20) if not is_bearish else (-count * 20)

                    fig.add_annotation(
                        x=date, y=y_val,
                        yshift=offset_y,
                        text=f"「{clean_name}」",
                        showarrow=False,
                        font=dict(color=font_color, size=12, weight="bold"),
                        row=1, col=1
                    )
                else:
                    # Single point pattern (Hammer, etc.): use arrow
                    # Increase ay to separate arrows
                    ay_val = (-30 - count*25) if is_bearish else (30 + count*25)

                    fig.add_annotation(
                        x=date, y=y_val,
                        text=clean_name,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor=font_color,
                        ax=0, ay=ay_val,
                        font=dict(color=font_color, size=12, weight="bold"),
                        row=1, col=1
                    )
"""
print("ok")
