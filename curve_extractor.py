import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np, cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import os
import io

st.session_state.update(st.session_state)
for k, v in st.session_state.items():
    st.session_state[k] = v
    
path = os.path.dirname(__file__)
#my_file = path + '/images/mechub_logo.png'
my_file = 'images/mechub_logo.png'
img_logo = Image.open(my_file)

st.set_page_config(
    page_title='Curve Extractor',
    layout="wide",
    page_icon=img_logo
)

hide_menu = '''
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        '''
st.markdown(hide_menu, unsafe_allow_html=True)

st.title("Curve Extractor V1.0.0: From Chart Image to Function", anchor=False)

tab_edit, tab_result, tab_about = st.tabs(
        ["Edit","Result","About"]
    )

sidebar = st.sidebar

uploaded = sidebar.file_uploader("Upload an image of a chart", type=["png","jpg","jpeg"])
mode = sidebar.radio("Mode", ["Default","Transparent BG"])
brush = sidebar.slider("Eraser size", 3, 80, 25)
pick = sidebar.color_picker("Background color (adjust manually)", "#FFFFFF")


#st.image(uploaded)
#st.write(st.__version__) funciona com streamlit_drawable_canvas na 1.40<

def autodetect_bg(img_rgb):
    h, w, _ = img_rgb.shape
    pts = np.vstack([
        img_rgb[0:30, 0:30].reshape(-1,3),
        img_rgb[0:30, w-30:w].reshape(-1,3),
        img_rgb[h-30:h, 0:30].reshape(-1,3),
        img_rgb[h-30:h, w-30:w].reshape(-1,3),
    ])
    return np.median(pts, axis=0).astype(np.uint8)

def get_graph_curve_points(img_name):
    # Carrega imagem em tons de cinza
    img_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    # Inverte se o fundo for claro
    if img_gray[0, 0] > 50:
        img_gray = cv2.bitwise_not(img_gray)

    # Binariza
    _, threshold = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)

    # Encontra todos os pixels do contorno (sem perder nenhum)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    all_points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            all_points.append((x, y))

    return all_points


def escalar_curva(contorno, x_fim, y_fim, x_ini=0., y_ini=0., eq_scale=True):
  if not contorno:
    return []

  xs, ys = zip(*contorno)

  min_x_px, max_x_px = min(xs), max(xs)
  min_y_px, max_y_px = min(ys), max(ys)

  escala_x = (x_fim - x_ini) / (max_x_px - min_x_px)

  if eq_scale:
    escala_y = escala_x
  else:
    escala_y = (y_fim - y_ini) / (max_y_px - min_y_px)

  curva_esc = [
    (x_ini + (x - min_x_px) * escala_x,
      y_ini + (y - min_y_px) * escala_y)
    for x, y in contorno
  ]

  return curva_esc

if uploaded:
    img = Image.open(uploaded).convert("RGBA")
    # Calculate the scaling factor
    width_ratio = 800 / img.width
    height_ratio = 800 / img.height
    scaling_factor = min(width_ratio, height_ratio)
    new_width = int(img.width * scaling_factor)
    new_height = int(img.height * scaling_factor)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    #max_size = (800, 800)
    #img = img.resize(max_size, Image.Resampling.LANCZOS)

    bg_img = np.array(img)  # RGBA
    rgb = bg_img[..., :3].copy()

    auto = sidebar.checkbox("Detect background color automatically", value=True)
    if auto:
        bgcol = autodetect_bg(rgb)
    else:
        # do colorpicker (#RRGGBB) para np.uint8
        bgcol = np.array([int(pick[i:i+2],16) for i in (1,3,5)], dtype=np.uint8)

    #col1.write("Background color:", bgcol.tolist())

    # X,Y Limits
    sidebar.divider()
    sidebar.subheader('Degree of The Polynomial')
    poly_dg = sidebar.number_input("> Degree of The Polynomial", value=4, label_visibility="hidden")
    sidebar.subheader('Chart x,y Limits')
    cols1, cols2 = sidebar.columns(2)
    x_ini = cols1.number_input("Initial X", value=0.)
    x_fim = cols1.number_input("Final X", value=1.)
    y_ini = cols2.number_input("Initial Y", value=0.)
    y_fim = cols2.number_input("Final Y", value=1.)

    # Canvas sobre a imagem (devolve as pinceladas)

    with tab_edit:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.0)",
            stroke_width=brush,
            stroke_color="#FF0000",
            background_image=img,
            update_streamlit=True,
            height=img.height,
            width=img.width,
            drawing_mode="freedraw",
            key="canvas",
        )

    if canvas_result.image_data is not None:
        # máscara de onde houve traço (alpha > 0)
        draw = (canvas_result.image_data[...,3] > 0).astype(np.uint8)*255  # 0/255

        if tab_edit.button("Apply",use_container_width=True):
            out = bg_img.copy()  # RGBA
            if mode=="Default":
                out[draw==255, 0] = bgcol[0]
                out[draw==255, 1] = bgcol[1]
                out[draw==255, 2] = bgcol[2]
                # mantém alpha original
            else:
                # apagar para transparente
                out[draw==255, 3] = 0

            #col2.image(out, use_container_width=True)
            # download
            out_bgra = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite("result_img.png", out_bgra)
            tab_edit.download_button("Donwload result_img.png", data=open("result_img.png","rb").read(), file_name="result_img.png",use_container_width=True)

            # Extrai os pontos
            points = get_graph_curve_points("result_img.png")

            # Separa listas X e Y
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            # Inverte Y (pois ficaria ao contrário)

            ys = np.max(ys) - ys

            points = list(zip(xs, ys))

            points = escalar_curva(points, x_fim, y_fim, x_ini, y_ini, False)

            xs, ys = zip(*points)

            with tab_result:
                col1, col2 = st.columns([1, 2])

            # Dados
            x = np.array(xs)
            y = np.array(ys)

            # Dicionário para armazenar modelos
            modelos = {}

            # Polinômios de grau 1 a 4
            for grau in range(poly_dg-2, poly_dg+3):
                coef = np.polyfit(x, y, grau)
                p = np.poly1d(coef)
                y_pred = p(x)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                modelos[f'Polynomial of degree {grau}'] = (p, y_pred, rmse)

                # Print da equação
                #eq = f'Polynomial of degree {grau}:\ny = '
                eq = ''
                for i, c in enumerate(coef):
                    pot = len(coef) - i - 1
                    if pot == 0:
                        # eq += f'({c:.6f})'
                        eq += f'({c})'
                    else:
                        # eq += f'({c:.6f}) * x ** {pot} + '
                        eq += f'({c}) * x ** {pot} + '
                with col1.expander(f'Polynomial of degree {grau}'):
                    st.write(eq)
                    #col1.write(f"RMSE: {rmse:.6f}\n")


            # grau = poly_dg
            # coef = np.polyfit(x, y, grau)
            # p = np.poly1d(coef)
            # y_pred = p(x)
            # rmse = np.sqrt(mean_squared_error(y, y_pred))
            # modelos[f'Polinômio grau {grau}'] = (p, y_pred, rmse)
            #
            # # Print da equação
            # #eq = f'Polinômio grau {grau}:\ny = '
            # eq = ''
            # for i, c in enumerate(coef):
            #     pot = len(coef) - i - 1
            #     if pot == 0:
            #         #eq += f'({c:.6f})'
            #         eq += f'({c})'
            #     else:
            #         #eq += f'({c:.6f}) * x ** {pot} + '
            #         eq += f'({c}) * x ** {pot} + '
            # with col1.expander(f'Polinômio grau {grau}'):
            #     st.write(eq)
            #     #col1.write(f"RMSE: {rmse:.6f}\n")


            # # Modelo exponencial: y = a * exp(bx)
            # def modelo_exp(x, a, b):
            #     return a * np.exp(b * x)
            #
            #
            # try:
            #     popt_exp, _ = curve_fit(modelo_exp, x, y, p0=(0.3, 0.001), maxfev=10000)
            #     y_exp = modelo_exp(x, *popt_exp)
            #     rmse_exp = np.sqrt(mean_squared_error(y, y_exp))
            #     modelos['Exponencial'] = (lambda x: modelo_exp(x, *popt_exp), y_exp, rmse_exp)
            #
            #     # Print da equação
            #     with col1.expander('Exponencial'):
            #         st.write(f"{popt_exp[0]:.6f} * exp({popt_exp[1]:.6f} * x)")
            #         #col1.write(f"RMSE: {rmse_exp:.6f}\n")
            # except:
            #     pass


            # # Modelo logarítmico: y = a * log(x) + b
            # def modelo_log(xx, a, b, c):
            #     """Modelo: y = a * ln(x + c) + b  — robusto para x <= 0"""
            #     return a * np.log(xx + c) + b


            # Plotagem
            fig = plt.figure(figsize=(10, 10))
            x_fit = np.linspace(min(x), max(x), 200)
            plt.scatter(x, y, label='Data', color='black')

            for nome, (func, y_model, erro) in modelos.items():
                try:
                    y_fit = func(x_fit)
                    plt.plot(x_fit, y_fit, label=f'{nome} (RMSE={erro:.5f})')
                except:
                    continue

            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            #plt.show()
            with tab_result:
                col2.pyplot(fig)
                # Ranking por RMSE
                col1.write("Ranking the functions by error (RMSE):")
                ranking = sorted(modelos.items(), key=lambda x: x[1][2])
                for nome, (_, __, erro) in ranking:
                    col1.write(f"{nome}: RMSE = {erro:.6f}")

with tab_about:
    st.markdown('''
    ## About Curve Extractor:

    **Curve Extractor** lets you upload any chart image or screenshot and turn it into usable data and a fitted mathematical function.
    
    You can erase axes, text, numbers, and noise using a simple brush tool, leaving only the curve.  
    The app then detects the curve pixels, converts them into XY points, and fits polynomial models to generate a clean mathematical representation.
    
    ## How It Works:
    
    1. **Upload an Image** of a plot or graph.  
    2. **Clean the Curve** by removing axes, labels, and unwanted elements.  
    3. **Extract Points** automatically from the remaining curve.  
    4. **Generate Polynomial Fits** and view equations and errors.  
    
    Use it to digitize curves from papers, PDFs, screenshots, or experimental plots quickly and accurately.

    ''')

st.sidebar.image(img_logo)
st.sidebar.markdown(
    "[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Mechub?sub_confirmation=1) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GitMechub)")




