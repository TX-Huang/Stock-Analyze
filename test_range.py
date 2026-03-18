import re

code = """
        colors = ['#ef4444' if row['Close'] >= row['Open'] else '#22c55e' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

        # Initial View Range (2 Months) & Max Range Limit
        # Calculate approx 60 trading days for 2 months
        view_len = min(60, len(df))
        start_date = df.index[-view_len]
        end_date = df.index[-1]

        # In Plotly, to set initial range and restrict max zoom out, we update the x-axis
        fig.update_xaxes(
            range=[start_date, end_date],
            minallowed=df.index[0],
            maxallowed=df.index[-1],
            row=1, col=1
        )
        # Apply to volume chart as well
        fig.update_xaxes(
            range=[start_date, end_date],
            minallowed=df.index[0],
            maxallowed=df.index[-1],
            row=2, col=1
        )

        fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
"""
print("ok")
