import io
import polars as pl
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

DEBUG = False  # Set to False in production
"""
Your gmm parquet should look something like this:
+----
| name | mode | feature | means | covars | weights |
+----+----+----+----+----+----+
| emitter_1 | 1 | freq | [1000., 1200.] | [[100., 110.]] | [0.5, 0.5] |
| emitter_1 | 1 | pw | [200., 210.] | [[4., 4.]] | [0.5, 0.5] |
| emitter_1 | 1 | pri | [300., 320.] | [[9., 9.]] | [0.5, 0.5] |
| emitter_1 | 2 | freq | [1100., 1300.] | [[120., 130.]] | [0.5, 0.5] |
| emitter_1 | 2 | pw | [220., 230.] | [[5., 5.]] | [0.5, 0.5] |
| emitter_1 | 2 | pri | [330., 340.] | [[16., 16.]] | [0.5, 0.5] |
+----+----+----+----+----+----+

Your log parquet should look something like this:
+----
| freq_means | freq_stddevs | pw_means | pw_stddevs | pri | pred |
+----+----+----+----+----+----+
| [1000., 1200.] | [100., 110.] | [200., 210.] | [4., 4.] | [300., 320.] | emitter_1_1 |
| [1100., 1300.] | [120., 130.] | [220., 230.] | [5., 5.] | [330., 340.] | emitter_1_2 |
+----+----+----+----+----+----+
"""

global row_idx, max_row
row_idx = 0

def highlight_row(row):
    """
    Highlights the correct row inside the 7-row displayed window.
    The selected row should always be in the **middle (index 3)**,
    except when near the start or end of the dataframe.
    """
    global row_idx, max_row
    
    start_idx = max(0, row_idx - 3)
    end_idx = min(max_row, row_idx + 3)

    # Compute adjusted row index within the sliced dataframe
    if row_idx < 3:  # Near the start
        adjusted_row_idx = row_idx  # Keep highlighting at the same index
    elif row_idx > max_row - 3:  # Near the end
        adjusted_row_idx = (end_idx - start_idx) - (max_row - row_idx)
    else:
        adjusted_row_idx = 3  # Center row

    return ['background-color: #FFB838' if row.name == adjusted_row_idx else '' for _ in row]

def get_range(means, stddevs, min_val=1000, max_val=10000):
	"""
	Auto-scale the x-axis range based on the provided means and standard deviations.
	A lower bound of 1000 and an upper bound of 10000 are enforced.
	"""
	x_min = min(means) - 4 * max(stddevs)
	x_min = min(x_min, min_val)
	x_max = max(means) + 4 * max(stddevs)
	x_max = max(x_max, max_val)
	return np.linspace(x_min, x_max, 500)


def plot_feature(feature, log_df, gmm_df, row_idx, colors):
	"""
	Generate a Plotly figure for the specified feature.
	
	For "freq" (frequency) and "pw" (pulsewidth), the LOG DataFrame is expected to have 
	mean and standard deviation arrays; bell curves are computed and plotted.
	
	For "pri" (PRI), the LOG DataFrame is expected to have raw measurement values.
	These are plotted as individual scatter points (without bell curves).
	
	In all cases, the matching GMM components (from the GMM DataFrame) are overlaid
	as dashed bell curves.
	"""
	# Create a new Plotly figure
	fig = go.Figure()
	
	# Process based on feature type
	if feature == "pri":
		# --- For PRI ("pri"): Plot raw measurement points ---
		raw_points_list = log_df["pri"].to_list()
		if not raw_points_list:
			st.error("No valid raw measurement values in LOG file for feature 'pri'.")
			return None
		# Select raw points from the chosen row (assumed to be a list of measurements)
		selected_points = np.array(raw_points_list[row_idx])
		# Plot raw points as scatter markers along a baseline (y=0)
		fig.add_trace(go.Scatter(
			x=selected_points,
			y=np.zeros_like(selected_points),
			mode="markers",
			marker=dict(color=colors[0], size=8),
			name=f"Raw Points (pri) for row {row_idx}"
		))
		# Define x_range for overlaying GMM components (extend a little beyond raw data)
		x_range = np.linspace(min(selected_points)-10, max(selected_points)+10, 500)
	else:
		# --- For "freq" (frequency) and "pw" (pulsewidth): Plot bell curves from LOG data ---
		if feature == "freq":
			means_col = "freq_means"
			stddevs_col = "freq_stddevs"
			range_min = 1000
			range_max = 10000
		elif feature == "pw":
			means_col = "pw_means"
			stddevs_col = "pw_stddevs"
			range_min = 5000
			range_max = 10000
		else:
			st.error("Unsupported feature provided.")
			return None

		means_list = log_df[means_col].to_list()
		stddevs_list = log_df[stddevs_col].to_list()
		
		if not means_list or not stddevs_list:
			st.error(f"No valid mean or standard deviation values in LOG file for feature {feature}.")
			return None
		
		selected_means = np.array(means_list[row_idx])
		selected_stddevs = np.array(stddevs_list[row_idx])
		x_range = get_range(selected_means, selected_stddevs, range_min, range_max)
		
		# Plot a bell curve for each (mean, stddev) pair from the LOG data
		for mi in range(len(selected_means)):
			y = (1 / (selected_stddevs[mi] * np.sqrt(2 * np.pi))) * \
				np.exp(-0.5 * ((x_range - selected_means[mi]) / selected_stddevs[mi]) ** 2)
			fig.add_trace(go.Scatter(
				x=x_range,
				y=y,
				mode="lines",
				line=dict(color=colors[0], width=2),
				name=f"LOG Row {row_idx} Mean {selected_means[mi]:.1f}"
			))
	
	# --- Process matching GMM DataFrame row for the given feature ---
	if all(col in gmm_df.columns for col in ["means", "covars", "weights"]):
		# Extract emitter and mode from the LOG prediction column
		row_pred = log_df["pred"][row_idx]
		pred_parts = row_pred.split("_")
		emitter = "_".join(pred_parts[:-1])
		mode = pred_parts[-1]
		st.info(f"Processing GMM: emitter: {emitter}, mode: {mode}, feature: {feature}")
		
		# Filter the GMM DataFrame for the matching row
		row_filter = (gmm_df["name"] == emitter) & \
					 (gmm_df["mode"] == str(mode)) & \
					 (gmm_df["feature"] == feature)
					 
		if row_filter.sum() == 0:
			st.warning(f"No matching GMM row found for feature '{feature}'.")
		elif row_filter.sum() > 1:
			st.warning(f"Multiple matching GMM rows found for feature '{feature}'.")
		elif row_filter.sum() == 1:
			filtered_df = gmm_df.filter(row_filter)
			# Extract GMM parameters: means, covariances, and weights
			selected_gmm_means = filtered_df["means"].to_numpy()[0]
			# Assume covars are stored as a nested list (e.g. [[cov1, cov2]])
			selected_gmm_covars = np.array(filtered_df["covars"].to_list()[0]).flatten()
			selected_gmm_weights = filtered_df["weights"].to_numpy()[0]
			# Convert each covariance to a standard deviation (elementwise)
			selected_gmm_stddevs = np.sqrt(selected_gmm_covars)
			
			# For non-"pri" features, re-compute x_range based on GMM parameters
			if feature != "pri":
				x_range = get_range(selected_means, selected_stddevs, range_min, range_max)
			# For each GMM component, plot its bell curve as a dashed line
			for i in range(len(selected_gmm_means)):
				y = selected_gmm_weights[i] * (1 / (selected_gmm_stddevs[i] * np.sqrt(2 * np.pi))) * \
					np.exp(-0.5 * ((x_range - selected_gmm_means[i]) / selected_gmm_stddevs[i]) ** 2)
				fig.add_trace(go.Scatter(
					x=x_range,
					y=y,
					mode="lines",
					line=dict(color=colors[(i+1) % len(colors)], width=2, dash="dash"),
					name=f"GMM Comp {i} (Mean {selected_gmm_means[i]:.1f})"
				))
	else:
		st.warning("GMM DF missing required columns: means, covars, or weights.")

	# Update the figure layout with a range slider for the x-axis
	fig.update_layout(
		title=f"Row {row_idx}: {feature.upper()} - {'Raw Points' if feature=='p' else 'Distributions'} & GMM Components",
		xaxis_title="X values",
		yaxis_title="Probability Density" if feature != "pri" else "Raw Count",
		template="plotly_white",
		showlegend=True,
		xaxis=dict(rangeslider=dict(visible=True))
	)
	return fig

# Main application function
def main():
	# Set up page configuration with -inspired theme details
	st.set_page_config(
		page_title="Parquet Viewer -  Themed",
		page_icon="🔵",
		layout="wide"
	)

	# Inject custom CSS for  colors
	custom_css = """
	<style>
	/* Main container background */
	.reportview-container {
		background-color: #F5F5F5;
	}
	/* Sidebar background */
	.sidebar .sidebar-content {
		background-color: #E1E1E1;
	}
	/* Button styling with  blue accents */
	.stButton > button {
		background-color: #003366;
		color: #FFFFFF;
		border: none;
	}
	</style>
	"""
	st.markdown(custom_css, unsafe_allow_html=True)

	st.title("Parquet Viewer with  Theme")
	
	# --- File Upload Section ---
	if not DEBUG:
		gmm_files = st.file_uploader(
			"Drop your GMM-parquet files here",
			type=["parquet"],
			accept_multiple_files=True,
		)
		log_files = st.file_uploader(
			"Drop your LOG-parquet files here",
			type=["parquet"],
			accept_multiple_files=True,
		)
		if not gmm_files or not log_files:
			st.warning("Please upload both GMM and LOG Parquet files.")
			return
	else:
		gmm_files = [io.BytesIO(b"")]
		log_files = [io.BytesIO(b"")]

	# --- Load DataFrames ---
	if DEBUG:
		gmm_df = pl.read_parquet("gmm_lib.parquet")
		log_df = pl.read_parquet("tracks.parquet")
	else:
		gmm_df = pl.read_parquet(io.BytesIO(gmm_files[0].getvalue()))
		log_df = pl.read_parquet(io.BytesIO(log_files[0].getvalue()))

	st.subheader("LOG DataFrame Preview")
	st.write(f"LOG DataFrame shape: {log_df.shape}")
	# st.write(log_df.head().to_pandas())
	# st.dataframe(log_df.head().style.apply(highlight_row, axis=1))
	# --- Row Selection ---
	global row_idx, max_row

	max_row = log_df.shape[0] - 1
	row_idx_a = st.slider("Select row to display", 1, max_row+1, 1)
	row_idx = row_idx_a - 1

	start_idx = max(0, row_idx - 3)  # Ensure start doesn't go below 0
	end_idx = min(log_df.shape[0], row_idx + 3 + 1)  # Ensure end doesn't go beyond last index
	styled_df = log_df.slice(start_idx, end_idx-start_idx).to_pandas().style.apply(highlight_row, axis=1)
	st.dataframe(styled_df)

	

	# Define a color palette using Plotly Express
	colors = px.colors.qualitative.Set1

	# --- Create Tabs for Frequency, Pulsewidth, and PRI ---
	tab1, tab2, tab3 = st.tabs(["Frequency (f)", "Pulsewidth (w)", "PRI (p)"])
	
	# Plot for Frequency ("freq")
	with tab1:
		fig_f = plot_feature("freq", log_df, gmm_df, row_idx, colors)
		if fig_f is not None:
			st.plotly_chart(fig_f, use_container_width=True)
	
	# Plot for Pulsewidth ("pw")
	with tab2:
		fig_w = plot_feature("pw", log_df, gmm_df, row_idx, colors)
		if fig_w is not None:
			st.plotly_chart(fig_w, use_container_width=True)
	
	# Plot for PRI ("pri")
	with tab3:
		fig_p = plot_feature("pri", log_df, gmm_df, row_idx, colors)
		if fig_p is not None:
			st.plotly_chart(fig_p, use_container_width=True)

if __name__ == "__main__":
	main()
