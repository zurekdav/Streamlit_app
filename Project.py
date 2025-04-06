import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import BytesIO, StringIO

st.set_page_config(page_title="Analýza měření", layout="wide")
st.title("Analýza měření")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'statistics_results' not in st.session_state:
    st.session_state.statistics_results = []
if 'indirect_results' not in st.session_state:
    st.session_state.indirect_results = []
if 'custom_vars' not in st.session_state:
    st.session_state.custom_vars = []
if 'var_values' not in st.session_state:
    st.session_state.var_values = {}
if 'df_history' not in st.session_state:
    st.session_state.df_history = []
if 'current_state_index' not in st.session_state:
    st.session_state.current_state_index = 0
if 'last_pasted_data' not in st.session_state:
    st.session_state.last_pasted_data = ''

# Nahrání dat
st.header("Nahrání dat")
st.markdown("""
            - Máte-li možnost, urpavte soubor ještě před jeho nahráním.
            - Nahráváte-li soubor CSV, používejte destinnou tečku a oddělovač čárku.
            - Nahrát můžete i oblast dat zkopírovanou z Excelu/Google Sheets. V tomto případě vyberte možnost "Zkopírovat data".
            - Upravte hlavičku do podoby **'měřená veličina' ['jednotka']** (např. tedy **v [m/s]**).
            - **V případě bezrozměrné veličiny stejně použijete hranatou závorku (např. exp(x) [-]). Hranatá závorka nesmí zůstat prázdná.**
            - Budete-li zapisovat jednotku jakkoli jinak, program **nebude fungovat**. (Např. zápis T (s) nebo T \s nebude fungovat.)
            - Pokud v zápisu veličiny používate čísla nebo matematické operace (např. 1/T, exp(x)...), dejte výraz do **kulaté závorky** (tedy např. **(1/T)[1/s]**).
            """)

# Create tabs for file upload and paste data
tab1, tab2 = st.tabs(["Nahrát soubor", "Zkopírovat data"])

with tab1:
    uploaded_file = st.file_uploader("Nahrajte soubor typu .xlsx, .xls nebo .csv", type=['xlsx', 'xls', 'csv'])

with tab2:
    st.markdown("""
                - Vložte data zkopírovaná z Excelu/Google Sheets.
                - Data by již defaulně měla být oddělana tabulátory.
                """)
    pasted_data = st.text_area("Vložte data ze schránky:", height=200)
    if pasted_data:
        try:
            # Only process if the data is different from current state
            if pasted_data != st.session_state.get('last_pasted_data', ''):
                # Try to read the data with different separators
                try:
                    # First try with tab separator
                    df_from_clipboard = pd.read_csv(StringIO(pasted_data), sep='\t', na_values=[''])
                except:
                    # If that fails, try with comma separator
                    df_from_clipboard = pd.read_csv(StringIO(pasted_data), sep=',', na_values=[''])
                
                # Convert to numeric and handle decimal points
                df_from_clipboard = df_from_clipboard.replace(',', '.', regex=True)
                df_from_clipboard = df_from_clipboard.apply(pd.to_numeric, errors='coerce')
                
                # Set the data in session state
                st.session_state.df = df_from_clipboard
                st.session_state.file_name = "pasted_data"
                st.session_state.statistics_results = []
                st.session_state.df_history = [st.session_state.df.copy()]
                st.session_state.current_state_index = 0
                st.session_state.last_pasted_data = pasted_data
                
                st.success("Data byla úspěšně načtena!")
        except Exception as e:
            st.error(f"Chyba při načítání dat: {str(e)}")

# Process uploaded file if any
if uploaded_file is not None:
    try:
        # Only update data if a new file is uploaded
        if uploaded_file.name != st.session_state.file_name:
            # Read the file based on its type
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                st.session_state.df = pd.read_excel(uploaded_file)
                st.session_state.df = st.session_state.df.replace(',', '.', regex=True)
                st.session_state.df = st.session_state.df.apply(pd.to_numeric, errors='coerce')
            else:
                st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.session_state.statistics_results = []
            
            # Reset history with the original state
            st.session_state.df_history = [st.session_state.df.copy()]
            st.session_state.current_state_index = 0
    except Exception as e:
        st.error(f"Chyba při čtení souboru: {str(e)}")

# Display data and editing options if we have data (either from file or clipboard)
if st.session_state.df is not None:
    # Display the data
    st.subheader("Nahraná data")
    data_display = st.dataframe(st.session_state.df)
    
    # Undo functionality - always show the button
    if st.button("↩️ Vrátit poslední změnu"):
        try:
            # Try to undo the last change
            if st.session_state.current_state_index > 0:
                st.session_state.current_state_index -= 1
                st.session_state.df = st.session_state.df_history[st.session_state.current_state_index].copy()
                st.rerun()
        except (IndexError, ValueError):
            # If there's an error, reset to initial state
            st.session_state.current_state_index = 0
            st.session_state.df = st.session_state.df_history[0].copy()
            st.rerun()
    
    # Row and Column deletion sections
    st.subheader("🗑️ Vymazání řádků a sloupců")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vymazání řádku**")
        st.markdown("""
                    - Zadejte index řádku, který chcete odstranit.
                    - **Pozor, první řádek má číslo 0.**
                    """)
        row_to_delete = st.number_input("Zadejte index řádku k odstranění:", min_value=0, max_value=len(st.session_state.df)-1, value=0)
        if st.button("Odstranit řádek"):
            # When adding a new state, remove any states after the current one (if user had gone back in history)
            if st.session_state.current_state_index < len(st.session_state.df_history) - 1:
                st.session_state.df_history = st.session_state.df_history[:st.session_state.current_state_index + 1]
            
            # Apply the change
            modified_df = st.session_state.df.drop(row_to_delete).reset_index(drop=True)
            
            # Add the new state and update the index
            st.session_state.df_history.append(modified_df.copy())
            st.session_state.current_state_index += 1
            st.session_state.df = modified_df
            
            data_display.dataframe(st.session_state.df)
            st.success(f"Řádek {row_to_delete} byl odstraněn!")
    
    with col2:
        st.markdown("**Vymazání sloupce**")
        st.markdown("""
                    - Vyberte sloupec, který chcete odstranit.
                    - Zkontrolujte, že jste správně vybrali sloupec před jeho odstraněním.
                    """)
        col_to_delete = st.selectbox("Zvolte sloupec k odstranění:", st.session_state.df.columns, key="delete_column")
        if st.button("Odstranit sloupec"):
            # When adding a new state, remove any states after the current one (if user had gone back in history)
            if st.session_state.current_state_index < len(st.session_state.df_history) - 1:
                st.session_state.df_history = st.session_state.df_history[:st.session_state.current_state_index + 1]
            
            # Apply the change
            modified_df = st.session_state.df.drop(columns=[col_to_delete])
            
            # Add the new state and update the index
            st.session_state.df_history.append(modified_df.copy())
            st.session_state.current_state_index += 1
            st.session_state.df = modified_df
            
            data_display.dataframe(st.session_state.df)
            st.success(f"Sloupec {col_to_delete} byl odstraněn!")
    
    # Column multiplication section
    st.subheader("Sekce pro převod jednotek a přejmenování sloupců")
    st.markdown("""
                - Pro převod jednotek zvolte odovídající faktor (násobek) a změňte název sloupce. Můžete použít i sčítání/odčítání. **Zachovejte formát jednotky v hranaté závorce.**
                - Pokud chcete změnit název sloupce, zadejte faktor pro násobení 1 (resp. hodnotu k přičtení 0) a zadejte nové jméno sloupce. (Lze použít i pro úpravu na požadovaný formát hlavičky.)
                - pro násobení číslem 10^n použijte zápis pomocí "e" (**např. 1e-19 = 10^-19**).
                - Bacha, když převádíte víc sloupců, změny se nepropisují hned. Kontrolujete název sloupce, který měníte.
                """)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_column = st.selectbox("Zvolte sloupec k převodu:", st.session_state.df.columns)
    with col2:
        operation = st.selectbox("Zvolte operaci:", ["Násobení", "Sčítání"])
        if operation == "Násobení":
            value_label = "Zadejte násobek (např. 1e-19):"
            default_value = "1.0"
        else:
            value_label = "Zadejte konstantu k přičtení:"
            default_value = "0.0"
        value_str = st.text_input(value_label, value=default_value)
        try:
            value = float(value_str)
        except ValueError:
            st.error("Zadejte platné číslo")
            value = 0.0
    with col3:
        new_name = st.text_input("Zadejte nový název sloupce:", value=selected_column)
    
    if st.button("Převést a přejmenovat"):
        # When adding a new state, remove any states after the current one (if user had gone back in history)
        if st.session_state.current_state_index < len(st.session_state.df_history) - 1:
            st.session_state.df_history = st.session_state.df_history[:st.session_state.current_state_index + 1]
        
        # Apply the changes
        modified_df = st.session_state.df.copy()
        if operation == "Násobení":
            modified_df[selected_column] = modified_df[selected_column] * value
        else:  # Sčítání
            modified_df[selected_column] = modified_df[selected_column] + value
        
        modified_df = modified_df.rename(columns={selected_column: new_name})
        
        # Add the new state and update the index
        st.session_state.df_history.append(modified_df.copy())
        st.session_state.current_state_index += 1
        st.session_state.df = modified_df
        
        data_display.dataframe(st.session_state.df)

    # New column calculation section
    st.divider()
    st.subheader("Výpočet nového sloupce")
    st.markdown("""
                - Zadejte název nového sloupce ve formátu **název[jednotka]** (např. v[m/s])
                - Před výpočtem se ujistěte, že hlavička je ve formátu **název[jednotka]** (např. v[m/s])
                - Zadejte funkční vztah pomocí názvů existujících sloupců (např. s/t). Pište jen názva veličin bez jednotky (pro sloupec s[m] zadejte s).
                - K zobrazení nápovědy k zadávání funkcí klikněte na šipku v horním levém rohu.
                """)
    
    col1, col2 = st.columns(2)
    with col1:
        new_column_name = st.text_input("Zadejte název nového sloupce:", placeholder="v[m/s]")
    with col2:
        formula = st.text_input("Zadejte funkční vztah:", placeholder="s/t")
    
    if st.button("Vypočítat nový sloupec"):
        try:
            # Define allowed functions
            allowed_functions = {
                'exp': np.exp,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'asin': np.arcsin,
                'acos': np.arccos,
                'atan': np.arctan,
                'sinh': np.sinh,
                'cosh': np.cosh,
                'tanh': np.tanh,
                'asinh': np.arcsinh,
                'acosh': np.arccosh,
                'atanh': np.arctanh,
                'log': np.log,
                'sqrt': np.sqrt,
                'abs': np.abs,
                'pi': np.pi,
                'ee': np.e
            }

            # Validate new column name format
            if not re.match(r'^\s*[^\[\]]+\s*\[[^\[\]]+\]\s*$', new_column_name):
                st.error("❌ Název sloupce musí být ve formátu název[jednotka] (např. v[m/s])")
            else:
                # Extract base column names (without units) from existing columns
                base_columns = {}
                duplicate_columns = {}
                for col in st.session_state.df.columns:
                    # Extract the base name (everything before the square bracket)
                    base_name = col.split('[')[0].strip()
                    if base_name in base_columns:
                        if base_name not in duplicate_columns:
                            duplicate_columns[base_name] = [base_columns[base_name]]
                        duplicate_columns[base_name].append(col)
                    else:
                        base_columns[base_name] = col
                
                # Check for duplicate column names
                if duplicate_columns:
                    error_message = "❌ Nalezeny duplicitní názvy sloupců:\n"
                    for base_name, columns in duplicate_columns.items():
                        error_message += f"- Sloupce {', '.join(columns)} mají stejný základní název '{base_name}'\n"
                    st.error(error_message)
                else:
                    # Extract column names from the formula
                    # First, remove all function names and constants from the formula
                    formula_without_functions = formula
                    for func in allowed_functions.keys():
                        formula_without_functions = re.sub(r'\b' + func + r'\s*\(', '', formula_without_functions)
                    
                    # Now extract column names from the modified formula
                    formula_columns = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula_without_functions)
                    
                    # Check if all columns exist and create mapping for eval
                    eval_mapping = {}
                    for col in formula_columns:
                        if col not in base_columns:
                            st.error(f"❌ Sloupec '{col}' neexistuje v databázi. Dostupné sloupce jsou: {', '.join(base_columns.keys())}")
                            st.stop()
                        eval_mapping[col] = base_columns[col]
                    else:
                        # When adding a new state, remove any states after the current one
                        if st.session_state.current_state_index < len(st.session_state.df_history) - 1:
                            st.session_state.df_history = st.session_state.df_history[:st.session_state.current_state_index + 1]
                        
                        # Create a copy of the dataframe
                        modified_df = st.session_state.df.copy()
                        
                        # Calculate new column using eval with proper column names
                        try:
                            # Create a mapping of base names to their full column names
                            eval_formula = formula
                            for base_name, full_name in eval_mapping.items():
                                # Replace the base name with the full column name in the formula
                                eval_formula = re.sub(r'\b' + base_name + r'\b', f"@modified_df['{full_name}']", eval_formula)
                            
                            # Calculate the new column using @ to reference DataFrame columns
                            modified_df[new_column_name] = modified_df.eval(eval_formula, engine='python', local_dict=allowed_functions)
                            
                            # Add the new state and update the index
                            st.session_state.df_history.append(modified_df.copy())
                            st.session_state.current_state_index += 1
                            st.session_state.df = modified_df
                            
                            data_display.dataframe(st.session_state.df)
                            st.success(f"Nový sloupec {new_column_name} byl úspěšně přidán!")
                            
                        except Exception as e:
                            st.error(f"❌ Chyba při výpočtu: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Chyba: {str(e)}")

    # Statistics section
    st.divider()
    st.subheader("Střední hodnota a chyba přímého měření")
    st.markdown("""
                - Program vypočítá střední hodnotu a chybu přímého měření pro zvolený sloupec.
                - Pro výpočet je nutné, aby zápis jednotky byl ve tvaru **text[text2]**.
                - Příklad: **T[s]**
                """)
    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        selected_stat_column = st.selectbox("Vyberte sloupec ke zpracování:", st.session_state.df.columns)
    with stat_col2:
        if st.button("Vyhodnotit"):
            # Check for duplicate column names
            base_columns = {}
            duplicate_columns = {}
            for col in st.session_state.df.columns:
                base_name = col.split('[')[0].strip()
                if base_name in base_columns:
                    if base_name not in duplicate_columns:
                        duplicate_columns[base_name] = [base_columns[base_name]]
                    duplicate_columns[base_name].append(col)
                else:
                    base_columns[base_name] = col
            
            if duplicate_columns:
                error_message = "❌ Nalezeny duplicitní názvy sloupců:\n"
                for base_name, columns in duplicate_columns.items():
                    error_message += f"- Sloupce {', '.join(columns)} mají stejný základní název '{base_name}'\n"
                st.error(error_message)
            else:
                # Validate header format
                header_row = st.session_state.df.columns
                valid_format = all(
                    pd.isna(cell) or (
                        isinstance(cell, str) and 
                        bool(re.match(r'^\s*[^\[\]]+\s*\[[^\[\]]+\]\s*$', cell))
                    )
                    for cell in header_row
                )
                if valid_format:
                    values = st.session_state.df[selected_stat_column].dropna().to_numpy()
                    n = len(values)
                    mean = sum(values) / n
                    std = np.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
                    sme = std / np.sqrt(n)
                    
                    result = {
                        'column': selected_stat_column,
                        'mean': mean,
                        'error': sme
                    }
                    
                    existing_indices = [i for i, r in enumerate(st.session_state.statistics_results) 
                                     if r['column'] == selected_stat_column]
                    
                    if existing_indices:
                        st.session_state.statistics_results[existing_indices[0]] = result
                    else:
                        st.session_state.statistics_results.append(result)
                else:
                    st.error("❌ Hlavička musí být ve formátu text[text2]. Příklad: T[s] nebo Time [seconds]")

    # Display statistics results
    if st.session_state.statistics_results:
        st.subheader("Střední hodnoty a chyby")
        for i, result in enumerate(st.session_state.statistics_results):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{result['column']} = {result['mean']:.10f} ± {result['error']:.10f}")
            with col2:
                if st.button("Odstranit", key=f"remove_{i}"):
                    st.session_state.statistics_results.pop(i)
                    st.rerun()

    # Data Fitting and Plotting section
    st.divider()
    st.subheader("📈 Fitování a grafické zobrazení dat")
    st.markdown("Chcetel-li přidat na osu y chybové úečky, mějte chybu veličiny v samostatném sloupci.")
    
    # Column selection for plotting
    plot_col1, plot_col2, plot_col3 = st.columns(3)
    with plot_col1:
        x_column = st.selectbox("Vyberte sloupec pro osu x:", st.session_state.df.columns, key="x_data")
    with plot_col2:
        y_column = st.selectbox("Vyberte sloupec pro osu y:", st.session_state.df.columns, key="y_data")
    with plot_col3:
        error_column = st.selectbox("Vyberte sloupec pro chybu y (volitelné):", 
                                  ["None"] + list(st.session_state.df.columns), 
                                  key="error_data")

    # Custom labels
    col1, col2 = st.columns(2)
    with col1:
        x_label = st.text_input("Popisek osy x (volitelné):", value="")
    with col2:
        y_label = st.text_input("Popisek osy y (volitelné):", value="")
    
    # Custom labels for data and fit
    col1, col2 = st.columns(2)
    with col1:
        data_label = st.text_input("Popisek dat v legendě (volitelné):", value="Data")
    with col2:
        fit_label = st.text_input("Rovnice fitu v legendě (můžete použít matematický zápis v Latexu, viz nápověda):", value="Fit")

                            

    # Fit function selection
    st.markdown("### Vyberte typ fitu")
    predefined_function = st.selectbox(
        "Vyberte typ fitu:",
        ["Lineární funkce (a*x + b)", 
         "Kvadratická funkce (a*x^2 + b*x + c)",
         "Mocninná funkce (a*x^b)",
         "Exponenciální funkce (a*exp(b*x))", 
         "Exponenciální klesající funkce (a*exp(-b*x))",
         "Logaritmická funkce (a*log(x) + b)"]
    )
    
    # Define the fitting functions
    def linear(x, a, b):
        return a * x + b
        
    def exponential(x, a, b):
        return a * np.exp(b * x)
        
    def power_law(x, a, b):
        return a * np.power(x, b)
        
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
        
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)
        
    def logarithmic(x, a, b):
        return a * np.log(x) + b

    # Map selection to function and initial guesses
    fit_functions = {
        "Lineární funkce (a*x + b)": (linear, [1, 0]),
        "Kvadratická funkce (a*x^2 + b*x + c)": (quadratic, [1, 1, 0]),
        "Mocninná funkce (a*x^b)": (power_law, [1, 1]),
        "Exponenciální funkce (a*exp(b*x))": (exponential, [1, 0.1]),
        "Exponenciální klesající funkce (a*exp(-b*x))": (exp_decay, [1, -0.1]),
        "Logaritmická funkce (a*log(x) + b)": (logarithmic, [1, 0])
    }
    fit_function, p0 = fit_functions[predefined_function]

    if st.button("Vykreslit data a fit"):
        try:
            # Get data
            x_data = st.session_state.df[x_column].values
            y_data = st.session_state.df[y_column].values
            
            # Check for invalid values
            if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
                st.error("Data obsahují NaN hodnoty. Prosím, zkontrolujte svá data.")
            elif np.any(np.isinf(x_data)) or np.any(np.isinf(y_data)):
                st.error("Data obsahují nekonečné hodnoty. Prosím, zkontrolujte svá data.")
            else:
                # Handle errors if provided
                if error_column != "None":
                    y_errors = st.session_state.df[error_column].values
                    if np.any(np.isnan(y_errors)) or np.any(np.isinf(y_errors)):
                        st.error("Chybové hodnoty obsahují neplatná čísla. Prosím, zkontrolujte svá chybová data.")
                    else:
                        sigma = y_errors
                else:
                    sigma = None

                # Perform the fit with better error handling
                try:
                    # If errors are provided, use them for weighted fit
                    if sigma is not None:
                        weights = 1 / (sigma**2)
                        weights = weights / np.max(weights)  # Normalize to avoid numerical issues
                    else:
                        weights = None

                    # Perform the fit
                    popt, pcov = curve_fit(fit_function, x_data, y_data, 
                                         p0=p0,
                                         sigma=weights,
                                         absolute_sigma=False,
                                         maxfev=5000)

                    # Calculate parameter errors safely
                    perr = np.zeros_like(popt)
                    for i in range(len(popt)):
                        try:
                            if pcov[i,i] > 0:  # Check for positive variance
                                perr[i] = np.sqrt(pcov[i,i])
                            else:
                                perr[i] = 0
                        except:
                            perr[i] = 0

                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot data points with error bars if available
                    if error_column != "None":
                        ax.errorbar(x_data, y_data, yerr=y_errors, fmt='o', label=data_label, markersize=4)
                    else:
                        ax.scatter(x_data, y_data, label=data_label, s=30)

                    # Plot fit
                    x_fit = np.linspace(min(x_data), max(x_data), 1000)
                    try:
                        y_fit = fit_function(x_fit, *popt)
                        ax.plot(x_fit, y_fit, 'r-', label=fit_label)
                    except Exception as e:
                        st.error(f"Chyba při vykreslování fit křivky: {str(e)}")
                    else:
                        # Add labels and title
                        if x_label:
                            ax.set_xlabel(x_label)
                        else:
                            ax.set_xlabel(x_column)
                        if y_label:
                            ax.set_ylabel(y_label)
                        else:
                            ax.set_ylabel(y_column)
                        ax.grid(True, alpha=0.3)
                        ax.legend()

                        # Display plot in Streamlit
                        st.pyplot(fig)

                        # Display fit parameters
                        st.markdown("### Fit Parameters")
                        param_names = ['a', 'b', 'c', 'd', 'e'][:len(popt)]
                        for param, err, name in zip(popt, perr, param_names):
                            st.write(f"{name} = {param:.6f} ± {err:.6f}")

                        # Add option to download plot
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        st.download_button(
                            label="Stáhnout graf",
                            data=buf.getvalue(),
                            file_name="fit_plot.png",
                            mime="image/png"
                        )

                except RuntimeError as e:
                    st.error("Fit se nedařilo. Zkuste jiné počáteční parametry nebo jinou funkci.")
                except ValueError as e:
                    st.error(f"Chyba během fitování: {str(e)}")

        except Exception as e:
            st.error(f"Chyba při fitování: {str(e)}")
            st.error("Prosím, zkontrolujte svá data a fit funkci.")

# Indirect Measurement section
st.divider()
st.subheader("Nepřímé měření")
st.markdown("""
            - Pro zpracování nepřímo měřené veličiny můžete použít výsledky získané v předchozím kroku nebo zadat nové hodnoty.
            """)

# Create list of available variables
available_vars = []
if st.session_state.df is not None:
    for result in st.session_state.statistics_results:
        match = re.match(r'^([^\[\]]+)\s*\[', result['column'])
        if match:
            available_vars.append(match.group(1).strip())

# Define variables section
st.markdown("#### Zadejte proměnné, které vystupují ve funkční závislosti")
st.markdown(""" 
            - Chcete-li použít vypočítané hodnoty, zaškrtněte příslušné políčko.
            - Program vypíše i obecný vzorec pro výpočet chyby.
            """)

# Add new custom variable
col1, col2 = st.columns([3, 1])
with col1:
    new_var_name = st.text_input("Přidejte novou proměnnou:")
with col2:
    if st.button("Přidat proměnnou", key="add_new_var"):
        if new_var_name and new_var_name not in st.session_state.custom_vars:
            st.session_state.custom_vars.append(new_var_name)
            st.rerun()

# Display all variables
all_vars = available_vars + st.session_state.custom_vars

if all_vars:
    st.markdown("#### Proměnné")
    for var in all_vars:
        st.markdown(f"**{var}**")
        col1, col2 = st.columns(2)
        
        with col1:
            if var in available_vars:
                use_calculated = st.checkbox("Použít vypočítané hodnoty", key=f"use_calc_{var}")
            else:
                use_calculated = False
        
        if use_calculated:
            stat_result = next((r for r in st.session_state.statistics_results 
                             if re.match(r'^([^\[\]]+)\s*\[', r['column']).group(1).strip() == var), None)
            if stat_result:
                st.session_state.var_values[var] = {
                    'mean': stat_result['mean'],
                    'error': stat_result['error']
                }
        else:
            with col2:
                custom_mean = st.number_input(f"Zadejte střední hodnotu pro {var}", key=f"mean_{var}")
                custom_error = st.number_input(f"Zadejte chybu pro {var}", key=f"error_{var}")
                st.session_state.var_values[var] = {
                    'mean': custom_mean,
                    'error': custom_error
                }
        
        if var in st.session_state.custom_vars:
            if st.button("Odstranit proměnnou", key=f"remove_custom_var_{var}"):
                st.session_state.custom_vars.remove(var)
                if var in st.session_state.var_values:
                    del st.session_state.var_values[var]
                st.rerun()
    
    # Define functional relationship section
    st.markdown("#### Definování funkční závislosti")
    st.markdown("**Nejprve zadejte symbol nepřímo měřené veličiny (např. *v* pro rychlost)**")
    quant_name = st.text_input("Zadejte název veličiny (např. *v*):")
    st.markdown("**Nyní zadejte funkční závislost. (např. s/t)**")
    st.markdown("""
                - Používeje jen proměnné (písmenka) definované výše. Nepište jednotky v hranaté závorce (např. pro rychlost zadejte v, nikoli v[m/s]).
                - Pro zobrazení nápovědy k zadávání funkcí klikněte na šipku v horním levém rohu.
                """)
    functional_relation = st.text_input("Zadejte funkční závislost (např. *s/t*):")
    if st.button("Vyhodnotit", key="calculate_indirect"):
        try:
            # Define allowed functions for sympy
            allowed_sympy_functions = {
                'exp': sp.exp,
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'asin': sp.asin,
                'acos': sp.acos,
                'atan': sp.atan,
                'sinh': sp.sinh,
                'cosh': sp.cosh,
                'tanh': sp.tanh,
                'asinh': sp.asinh,
                'acosh': sp.acosh,
                'atanh': sp.atanh,
                'log': sp.log,
                'sqrt': sp.sqrt,
                'abs': sp.Abs,
                'pi': sp.pi,
                'ee': sp.E
            }
            
            # Extract variables from the functional relation
            # Extract all identifiers (words)
            all_identifiers = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', functional_relation))
            
            # Remove function names from the list of variables
            used_vars = all_identifiers - set(allowed_sympy_functions.keys())
            
            # Filter out only variables that are actually defined
            defined_vars = [var for var in used_vars if var in st.session_state.var_values]
            
            if not defined_vars:
                st.error("❌ Žádné definované proměnné nebyly nalezeny ve funkčním vztahu.")
                st.stop()
                
            if len(used_vars) != len(defined_vars):
                undefined_vars = used_vars - set(defined_vars)
                st.error(f"❌ Následující proměnné nejsou definovány: {', '.join(undefined_vars)}")
                st.stop()
            
            # Create symbols for variables
            symbol_mapping = {var: sp.Symbol(var) for var in defined_vars}
            
            try:
                # Create the functional relationship
                # Use a safer approach with explicit sympy parsing
                expr_text = functional_relation
                
                # Replace with safer explicit substitutions
                for func_name, func in allowed_sympy_functions.items():
                    if func_name in expr_text and func_name not in ['pi', 'ee']:
                        # Handle special cases for constants
                        continue
                
                # Parse expression with allowed functions - with better error handling
                try:
                    f = parse_expr(expr_text, local_dict={**symbol_mapping, **allowed_sympy_functions}, transformations='all')
                except Exception as parse_error:
                    st.error(f"❌ Chyba při analýze výrazu: {str(parse_error)}")
                    st.error("Ujistěte se, že používáte platnou syntaxi a závorky jsou správně uzavřeny.")
                    st.stop()
                
                # Calculate mean value with better error handling
                try:
                    values_dict = {var: v['mean'] for var, v in st.session_state.var_values.items() if var in defined_vars}
                    f_mean = f.subs(values_dict).evalf()
                except Exception as eval_error:
                    st.error(f"❌ Chyba při výpočtu střední hodnoty: {str(eval_error)}")
                    st.stop()
                
                # Calculate derivatives and errors with better error handling
                try:
                    derivatives = []
                    evaluated_derivatives = []
                    for var in defined_vars:
                        try:
                            derivative = sp.diff(f, symbol_mapping[var])
                            evaluated_derivative = derivative.subs(values_dict).evalf()
                            derivatives.append(derivative)
                            evaluated_derivatives.append(evaluated_derivative)
                        except Exception as diff_error:
                            st.warning(f"Varování: Chyba při výpočtu derivace podle {var}: {str(diff_error)}")
                            derivatives.append(sp.sympify(0))
                            evaluated_derivatives.append(0)
                except Exception as deriv_error:
                    st.error(f"❌ Chyba při výpočtu derivací: {str(deriv_error)}")
                    st.stop()
                
                # Calculate indirect error
                indirect_error = sum([(st.session_state.var_values[var]['error'] * deriv) ** 2 
                                    for var, deriv in zip(defined_vars, evaluated_derivatives)]) ** 0.5
                
                # Display the error calculation formula
                st.markdown("**Vzorec pro výpočet chyby:**")
                
                # Use a safer approach for LaTeX rendering
                try:
                    latex_expr = functional_relation
                    # Create symbols for the variables with better approach
                    sym_vars = {var: sp.Symbol(var) for var in defined_vars}
                    
                    # Parse the expression for LaTeX display
                    try:
                        expr = parse_expr(latex_expr, local_dict={**sym_vars, **allowed_sympy_functions}, transformations='all')
                    except:
                        # Fallback if the parse fails for LaTeX
                        expr = f
                    
                    # Create the error formula with safer approach
                    try:
                        error_terms = []
                        for var in defined_vars:
                            try:
                                derivative = sp.diff(expr, sym_vars[var])
                                error_terms.append(f"({sp.latex(derivative)} \\cdot σ_{{{var}}})^2")
                            except:
                                error_terms.append(f"(\\frac{{∂f}}{{∂{var}}} \\cdot σ_{{{var}}})^2")
                        
                        error_formula = f"σ_{{{quant_name}}} = \\sqrt{{" + " + ".join(error_terms) + "}"
                        st.latex(error_formula)
                    except Exception as latex_error:
                        # Simplified fallback for error formula
                        st.markdown(f"Chyba při zobrazení vzorce: {str(latex_error)}")
                        st.markdown("Obecný vzorec pro chybu nepřímého měření: σ_f = √[Σ(∂f/∂x_i · σ_i)²]")
                except Exception as formula_error:
                    st.warning(f"Varování: Nelze vykreslit vzorec: {str(formula_error)}")
                
                # Create the result with appropriate handling of complex numbers
                try:
                    mean_value = complex(f_mean)
                    if mean_value.imag == 0:
                        result_mean = float(mean_value.real)
                    else:
                        st.warning("Výsledek obsahuje komplexní čísla. Zobrazuje se pouze reálná část.")
                        result_mean = float(mean_value.real)
                except:
                    # Fallback for non-numeric results
                    result_mean = float(f_mean)
                
                result = {
                    'name': quant_name,
                    'mean': result_mean,
                    'error': float(indirect_error)
                }
                
                existing_indices = [i for i, r in enumerate(st.session_state.indirect_results) 
                                if r['name'] == quant_name]
                if existing_indices:
                    st.session_state.indirect_results[existing_indices[0]] = result
                else:
                    st.session_state.indirect_results.append(result)
            except Exception as e:
                st.error(f"❌ Chyba v výpočtu: {str(e)}")
                st.error("Prosím, zkontrolujte, zda jsou všechny proměnné definovány a vzorec je správný.")
        except Exception as e:
            st.error(f"Chyba v výpočtu: {str(e)}")
            st.error("Prosím, zkontrolujte, zda jsou všechny proměnné definovány a vzorec je správný.")
    
    # Display indirect measurement results
    if st.session_state.indirect_results:
        st.markdown("### Výsledky nepřímého měření")
        for i, result in enumerate(st.session_state.indirect_results):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{result['name']} = {result['mean']:.10f} ± {result['error']:.10f}")
            with col2:
                if st.button("Odstranit", key=f"remove_indirect_{i}"):
                    st.session_state.indirect_results.pop(i)
                    st.rerun()
else:
    st.info("Prosím, přidejte alespoň jednu proměnnou.")

# Add sidebar with help text
with st.sidebar:
    st.markdown("### Nápověda")
    
    with st.expander("Nápověda pro výpočty", expanded=True):
        st.markdown("""
        - **!!!!!!!!!!!!Ve všech výpočtech pužívejte destinnou tečku místo desetinné čárky.!!!!!!!!!!!!**
        - Použijte `pi` pro π (např. `2*pi*r` pro obvod)
        - Použijte `ee` pro Eulerovo číslo
        - Použijte `**` pro umocnění (např. `r**2` pro r na druhou, `x**(1/3)` pro třetí odmocninu z x)
        - Použijte standardní operátory: `+`, `-`, `*`, `/`
        - Použijte závorky: `(a + b) * c`

        Matematické funkce:
        
        - `abs(x)` pro absolutní hodnotu **(nefunguje u nepřímého měření)**
        - `exp(x)` pro e^x
        - `sin(x)`, `cos(x)`, `tan(x)` pro goniometrické funkce
        - `asin(x)`, `acos(x)`, `atan(x)`  pro inverzní goniometrické funkce
        - `sinh(x)`, `cosh(x)`, `tanh(x)` pro hyperbolické funkce
        - `asinh(x)`, `acosh(x)`, `atanh(x)` pro inverzní hyperbolické funkce
        - `log(x)` pro přirozený logaritmus
        - `sqrt(x)` pro druhou odmocninu
        """)
    
    with st.expander("Nápověda pro matematický zápis v LaTeXu", expanded=True):
        st.markdown("""
- Všechny matematické výrazy uzavřete mezi `$` (např. `$y = kx + q$`)
- Násobení zapište pomocí `\cdot` (např. `$a \cdot b$`)
- Zlomky zapište pomocí `\\frac{čitatel}{jmenovatel}`
- Mocniny a horní indexy zapište pomocí `^` (např. `$x^2$`)
- Spodní indexy zapište pomocí `_` (např. `$x_1$`)
- Pokud mocnina nebo index obsahuje více než jedno písmeno, uzavřete je do složené závorky `{}` (např. `$x^{23}$` pro x na dvacátou třetí)
- Odmocniny zapište pomocí `\sqrt{vyraz}`
- Lineární funkci zapište jako `$y = k\cdot x + q$`
- π zapište jako `\pi`
 """)
