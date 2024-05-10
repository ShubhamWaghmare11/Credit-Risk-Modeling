import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title and description
st.set_page_config(page_title="credit Approval Predictor", page_icon=":money_with_wings:",layout='wide')



st.title("Credit Approval Predictor")
st.write("The model predicts the likelihood that a borrower will fail on a credit and classifies them in 4 categories (P1,P2,P3,P4),\nsegregating the customer into classes got given credit. \n\nP1 is the category where the bank can easily give credit to the customer, whereas P4 is the category where it is not a good idea to give credit to that customer, as it can increase the NPA accounts (non-Performing assets) of the bank.") 
st.write("Upload an Excel file containing borrower information to predict credit approval.")

st.subheader("")
st.write("Note: Make sure the excel file contains following columns with defined values and the column names are same as the names in Variable Name: ")
st.table(pd.read_excel(r"info.xlsx"))
# Load the trained model
model = pd.read_pickle('model.pkl')

# Define column names
cols_in_df = ['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M', 'Tot_Missed_Pmnt', 
              'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL', 'Age_Oldest_TL', 'Age_Newest_TL', 
              'time_since_recent_payment', 'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'num_times_60p_dpd', 
              'num_std_12mts', 'num_sub', 'num_sub_6mts', 'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss', 
              'recent_level_of_deliq', 'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 
              'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever', 
              'pct_CC_enq_L6m_of_ever', 'HL_Flag', 'GL_Flag', 'MARITALSTATUS', 'EDUCATION', 'GENDER', 
              'last_prod_enq2', 'first_prod_enq2']

cols_in_df = [col.lower() for col in cols_in_df]
st.set_option('deprecation.showPyplotGlobalUse', False)
# Function to preprocess and predict loan approval

def plot_graphs(df):
    col1,col2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 5.20))
    palette_color = sns.color_palette("deep")

    with col1:
            st.subheader("Distribution of Predicted Outcomes: ")
            plt.figure()
            plt.pie(df['Approved Flag'].value_counts(), labels=df['Approved Flag'].value_counts().index,autopct='%1.1f%%',colors=palette_color)
            plt.axis('equal')
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("Gender distribution in the data: ")
        
            plt.figure()
            plt.pie(df['GENDER'].value_counts(),labels=df['GENDER'].value_counts().index, autopct='%1.1f%%',colors=palette_color)
            plt.axis('equal')
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("Education distribution in the data: ")
            
            plt.figure()
            plt.pie(df['EDUCATION'].value_counts(),labels=df['EDUCATION'].value_counts().index,autopct='%1.1f%%',colors=palette_color)
            plt.axis('equal')
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("Distribution of Last product enquired for:")
            plt.figure()
            palette_color = sns.color_palette("deep")
            plt.pie(df['last_prod_enq2'].value_counts(),labels=df['last_prod_enq2'].value_counts().index,colors=palette_color,autopct='%.0f%%')
            plt.axis('equal')
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("Distribution of first product enquired for: ")
        
            plt.figure()
            plt.pie(df['first_prod_enq2'].value_counts(),labels=df['first_prod_enq2'].value_counts().index,colors=palette_color,autopct='%.0f%%')
            plt.axis('equal')
            st.pyplot()

    with col2:
            st.subheader("")
            st.subheader("")
            plt.figure()
            sns.barplot(df['Approved Flag'].value_counts())
            st.pyplot()

            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            plt.figure()
            sns.barplot(df['GENDER'].value_counts())
            st.pyplot()

            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            
            
            plt.figure()
            sns.barplot(df['EDUCATION'].value_counts(),orient='h',ax=ax)
            st.pyplot(fig)

            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            plt.figure()
            palette_color = sns.color_palette("deep")
            sns.barplot(df['last_prod_enq2'].value_counts())
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader('')
            st.subheader("")
        
            plt.figure()
            palette_color = sns.color_palette("deep")
            sns.barplot(df['first_prod_enq2'].value_counts())
            st.pyplot()    

def predict_loan_approval(file):
    try:
        # Read the uploaded Excel file
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df2 = df.copy()
        # Preprocess the data
        df2.columns = df2.columns.str.lower()
        df2 = df2[cols_in_df]
        for col in df2.select_dtypes(include='object'):
            df2[col] = df2[col].str.lower() 

        df2['education'] = df2['education'].map({"ssc":5, "10th":5, "12th":0, "graduate":1, "under graduate":6, 
                                                "post-graduate":3, "others":2, "professional":4})
        df2['education'] = df2['education'].astype(int)

        # Perform one-hot encoding
        df_encoded = pd.get_dummies(df2, columns=['maritalstatus', 'gender', 'last_prod_enq2', 'first_prod_enq2'])
        df_encoded_unseen = df_encoded[['pct_tl_open_l6m', 'pct_tl_closed_l6m', 'tot_tl_closed_l12m', 'pct_tl_closed_l12m', 'tot_missed_pmnt', 'cc_tl', 'home_tl', 'pl_tl', 'secured_tl', 'unsecured_tl', 'other_tl', 'age_oldest_tl', 'age_newest_tl', 'time_since_recent_payment', 'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_6mts', 'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss', 'recent_level_of_deliq', 'cc_enq_l12m', 'pl_enq_l12m', 'time_since_recent_enq', 'enq_l3m', 'netmonthlyincome', 'time_with_curr_empr', 'cc_flag', 'pl_flag', 'pct_pl_enq_l6m_of_ever', 'pct_cc_enq_l6m_of_ever', 'hl_flag', 'gl_flag', 'education', 'maritalstatus_married', 'maritalstatus_single', 'gender_f', 'gender_m', 'last_prod_enq2_al', 'last_prod_enq2_cc', 'last_prod_enq2_consumerloan', 'last_prod_enq2_hl', 'last_prod_enq2_pl', 'last_prod_enq2_others', 'first_prod_enq2_al', 'first_prod_enq2_cc', 'first_prod_enq2_consumerloan', 'first_prod_enq2_hl', 'first_prod_enq2_pl', 'first_prod_enq2_others']]
        # Make predictions
        predictions = model.predict(df_encoded_unseen)
        
        # Map prediction labels
        prediction_labels = {1:'P2', 2:'P3', 3:'P4', 0:'P1'}
        df['Approved Flag'] = [prediction_labels[prediction] for prediction in predictions]
        
        # Download the predicted results as an Excel file
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        excel_data = excel_buffer.getvalue()
        
        # Display download button for the Excel file
        st.download_button(label="Download Predicted Results", data=excel_data, file_name='predicted_results.xlsx', 
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                           help="Click to download the predicted results as an Excel file.")
        
        # Display success message
        st.success("Credit approval predictions generated successfully!")
        #st.write(df['Target_Variable'])

        plot_graphs(df)

        

    except Exception as e:
        st.error(f"An error occurred: {e}")

# File uploader widget
uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx","csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Perform loan approval prediction
    predict_loan_approval(uploaded_file)


