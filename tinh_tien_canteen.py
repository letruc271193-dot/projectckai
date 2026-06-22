import streamlit as st
import cv2
import json
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Hệ Thống Thanh Toán Canteen Tự Động", layout="wide")
st.title("🍔 AI Canteen Billing System")
st.markdown("Hệ thống tự động nhận diện món ăn, tính tiền, đo Calo và xuất mã QR thanh toán.")
@st.cache_resource
def load_assets():
    with open('menu.json', 'r', encoding='utf-8') as f:
        menu_data = json.load(f)
    with open('labels_cnn.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    labels = {v: k for k, v in class_indices.items()}
    model = load_model('model_efficientnet_pro.h5')
    return menu_data, labels, model

menu_data, class_labels, model = load_assets()

if "toa_do_khay" not in st.session_state:
    st.session_state.toa_do_khay = {
        'Ngan_1': (372,263,326,331),
        'Ngan_2': (752,119,396,318),
        'Ngan_3': (1225,132,271,322),
        'Ngan_4': (1031,608,396,319),
        'Ngan_5': (457,586,342,327)
    }

def chay_ai_nhan_dien(image_np):
    ket_qua_ai = []
    for ten_ngan, (x, y, w, h) in st.session_state.toa_do_khay.items():
        y_end = min(y+h, image_np.shape[0])
        x_end = min(x+w, image_np.shape[1])
        anh_cat = image_np[y:y_end, x:x_end]
        if anh_cat.shape[0] > 0 and anh_cat.shape[1] > 0:
            anh_resize = cv2.resize(anh_cat, (128, 128))
            anh_dua_vao = img_to_array(anh_resize)
            anh_dua_vao = np.expand_dims(anh_dua_vao, axis=0)
            anh_dua_vao = preprocess_input(anh_dua_vao)
            preds = model.predict(anh_dua_vao, verbose=0)
            xac_suat = float(np.max(preds))
            prediction = np.argmax(preds)
            ten_mon = class_labels[prediction]
        else:
            ten_mon = "Lỗi khung cắt"
            xac_suat = 0.0
            anh_cat = np.zeros((128, 128, 3), dtype=np.uint8)
        ket_qua_ai.append({
            "Ngăn": ten_ngan,
            "Tên Món AI": ten_mon,
            "Xác Suất": xac_suat,
            "Ảnh Cắt": anh_cat
        })
    return ket_qua_ai

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 Đầu vào hình ảnh")
    loai_dau_vao = st.radio("Chọn nguồn ảnh:", ("Chụp từ Camera", "Tải ảnh từ máy"))
    img_file = None
    if loai_dau_vao == "Chụp từ Camera":
        img_file = st.camera_input("Đưa khay cơm vào camera")
    else:
        img_file = st.file_uploader("Chọn file ảnh khay cơm", type=['jpg', 'jpeg', 'png'])

    if img_file is not None:
        image = Image.open(img_file)
        image_np = np.array(image)
        img_h, img_w = image_np.shape[0], image_np.shape[1]

        # CHỨC NĂNG TỰ ĐỘNG LƯU ẢNH KHAY GỐC
        if "last_saved_full_image" not in st.session_state:
            st.session_state.last_saved_full_image = None
            
        if img_file != st.session_state.last_saved_full_image:
            st.session_state.last_saved_full_image = img_file
            thu_muc_goc = "dataset_anh_khay_goc"
            os.makedirs(thu_muc_goc, exist_ok=True)
            
            thoi_gian_chup = datetime.now().strftime('%Y%m%d_%H%M%S')
            ten_file_goc = f"khay_goc_{thoi_gian_chup}.jpg"
            duong_dan_goc = os.path.join(thu_muc_goc, ten_file_goc)
            
            if image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            elif image_np.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = image_np
                
            cv2.imwrite(duong_dan_goc, image_bgr)
            st.toast("📸 Đã lưu ảnh khay gốc thành công!", icon="✅")

        st.markdown("---")
        with st.expander("🛠️ CÔNG CỤ: Chỉnh Khung Cắt Ảnh (Tọa độ)", expanded=True):
            st.info("Kéo thanh trượt để dời/phóng to khung xanh bên phải cho khớp với món ăn. Sau đó bấm nút Quét lại.")
            ten_ngan_tieng_viet = {k: k.replace('Ngan_', 'Ngăn ') for k in st.session_state.toa_do_khay.keys()}
            ngan_chon = st.radio(
                "Chọn Ngăn cần chỉnh:",
                options=list(st.session_state.toa_do_khay.keys()),
                format_func=lambda x: ten_ngan_tieng_viet.get(x, x),
                horizontal=True
            )
            x, y, w, h = st.session_state.toa_do_khay[ngan_chon]
            col_x, col_y = st.columns(2)
            new_x = col_x.slider("Sang trái/phải (X)", 0, img_w, x, key=f"x_{ngan_chon}")
            new_y = col_y.slider("Lên/xuống (Y)", 0, img_h, y, key=f"y_{ngan_chon}")
            col_w, col_h = st.columns(2)
            new_w = col_w.slider("Độ rộng (W)", 10, img_w, w, key=f"w_{ngan_chon}")
            new_h = col_h.slider("Độ cao (H)", 10, img_h, h, key=f"h_{ngan_chon}")
            st.session_state.toa_do_khay[ngan_chon] = (new_x, new_y, new_w, new_h)
            
            if st.button("🔄 Nhận diện lại với khung tọa độ mới", use_container_width=True, type="primary"):
                st.session_state.last_image = None
                st.rerun()

        st.markdown("### 🖼️ Hình ảnh các món ăn đã cắt")
        st.caption("Khung cắt sẽ cập nhật trực tiếp (live) khi bạn kéo thanh trượt ở trên.")
        cols_cat = st.columns(5)
        for i, (ten_ngan, (nx, ny, nw, nh)) in enumerate(st.session_state.toa_do_khay.items()):
            ny_end = min(ny+nh, img_h)
            nx_end = min(nx+nw, img_w)
            anh_cat_preview = image_np[ny:ny_end, nx:nx_end]
            with cols_cat[i]:
                ten_ngan_dep = ten_ngan.replace('Ngan_', 'Ngăn ')
                st.markdown(f"<p style='text-align: center; font-weight: bold; margin-bottom: 5px;'>{ten_ngan_dep}</p>", unsafe_allow_html=True)
                if anh_cat_preview.shape[0] > 0 and anh_cat_preview.shape[1] > 0:
                    st.image(anh_cat_preview, use_container_width=True)
                else:
                    st.error("Lỗi khung")

with col2:
    st.subheader("🧾 Quản lý Hóa Đơn")
    if img_file is not None:
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        danh_sach_menu = ["-- Bỏ trống (Không có món) --"] + list(menu_data.keys())
        tong_tien_goc = 0
        tong_phu_thu = 0
        tong_calo = 0
        chi_tiet_bill = []
        chi_tiet_luu_excel = []
        
        if "last_image" not in st.session_state:
            st.session_state.last_image = None
            st.session_state.ket_qua_ai = []
            
        if img_file != st.session_state.last_image:
            st.session_state.last_image = img_file
            with st.spinner('AI đang quét khay cơm...'):
                st.session_state.ket_qua_ai = chay_ai_nhan_dien(image_np)
                for item in st.session_state.ket_qua_ai:
                    ngan = item["Ngăn"]
                    ten_mon_ai = item["Tên Món AI"]
                    if ten_mon_ai in danh_sach_menu:
                        st.session_state[f"select_{ngan}"] = ten_mon_ai
                    else:
                        st.session_state[f"select_{ngan}"] = "-- Bỏ trống (Không có món) --"

        img_preview = image_np.copy()
        for ten_ngan, (x, y, w, h) in st.session_state.toa_do_khay.items():
            mon_hien_tai = st.session_state.get(f"select_{ten_ngan}", None)
            ai_xac_suat = 0
            ten_mon_ai = ""
            if "ket_qua_ai" in st.session_state:
                for item in st.session_state.ket_qua_ai:
                    if item["Ngăn"] == ten_ngan:
                        ai_xac_suat = item["Xác Suất"]
                        ten_mon_ai = item["Tên Món AI"]
                        if not mon_hien_tai:
                            mon_hien_tai = ten_mon_ai
                        break
                        
            if mon_hien_tai == ten_mon_ai or (mon_hien_tai == "-- Bỏ trống (Không có món) --" and ten_mon_ai not in danh_sach_menu):
                hien_thi_xac_suat = ai_xac_suat * 100
            else:
                hien_thi_xac_suat = 100.0

            if mon_hien_tai and mon_hien_tai != "-- Bỏ trống (Không có món) --":
                ten_khong_dau = menu_data.get(mon_hien_tai, {}).get("ten_khong_dau", mon_hien_tai.replace('_', ' '))
                text_to_show = f"{ten_khong_dau} ({hien_thi_xac_suat:.0f}%)"
            else:
                text_to_show = ten_ngan.replace('Ngan_', 'Ngan ')
                
            cv2.rectangle(img_preview, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text_to_show, font, font_scale, thickness)
            
            if int(y) - text_h - 10 < 0:
                bg_y1 = int(y)
                bg_y2 = int(y) + text_h + 10
                text_y = bg_y2 - 5
            else:
                bg_y1 = int(y) - text_h - 10
                bg_y2 = int(y)
                text_y = bg_y2 - 5
                
            cv2.rectangle(img_preview, (int(x), bg_y1), (int(x) + text_w + 10, bg_y2), (0, 255, 0), -1)
            cv2.putText(img_preview, text_to_show, (int(x) + 5, text_y), font, font_scale, (0, 0, 0), thickness)
            
        st.image(img_preview, caption="Khung Cắt AI & Tem Nhận Diện", use_container_width=True)
        st.markdown("### 🔍 Hiệu Chỉnh Khay Cơm")
        
        for item in st.session_state.ket_qua_ai:
            ngan = item["Ngăn"]
            ten_mon_ai = item["Tên Món AI"]
            xac_suat = item["Xác Suất"]
            mon_hien_tai = st.session_state.get(f"select_{ngan}", ten_mon_ai)
            
            if mon_hien_tai == ten_mon_ai or (mon_hien_tai == "-- Bỏ trống (Không có món) --" and ten_mon_ai not in danh_sach_menu):
                hien_thi_xac_suat = xac_suat * 100
            else:
                hien_thi_xac_suat = 100.0
            
            st.markdown(f"**📍 Vị trí: {ngan.replace('Ngan_', 'Ngăn ')}**")
            mon_chon_tay = st.selectbox(
                f"Độ tự tin: {hien_thi_xac_suat:.1f}%",
                options=danh_sach_menu,
                format_func=lambda x: menu_data.get(x, {}).get("ten_tieng_viet", x),
                key=f"select_{ngan}",
                label_visibility="collapsed"
            )
            
            if mon_chon_tay != "-- Bỏ trống (Không có món) --":
                ten_mon_chot = mon_chon_tay
            else:
                ten_mon_chot = None
                
            if ten_mon_chot is not None:
                gia_goc = menu_data.get(ten_mon_chot, {}).get("gia", 0)
                calo = menu_data.get(ten_mon_chot, {}).get("calo", 0)
                phu_thu = 0
                ghi_chu = ""
                
                if ten_mon_chot == "Thit_kho_trung":
                    so_trung_them = st.number_input(f"🥚 Thêm trứng cho {ngan.replace('Ngan_', 'Ngăn ')}? (+6,000 VNĐ)", min_value=0, max_value=5, value=0, step=1, key=f"trung_{ngan}")
                    if so_trung_them > 0:
                        phu_thu = so_trung_them * 6000
                        calo += so_trung_them * 78
                        ghi_chu = f"+{so_trung_them} trứng"
                        
                if ten_mon_chot == "Com_trang":
                    goi_com_them = st.checkbox(f"🍚 Cơm thêm cho {ngan.replace('Ngan_', 'Ngăn ')}? (+2,000 VNĐ)", key=f"com_{ngan}")
                    if goi_com_them:
                        phu_thu = 2000
                        calo += 130
                        ghi_chu = "+Cơm thêm"

                thanh_tien_mon = gia_goc + phu_thu
                tong_tien_goc += gia_goc
                tong_phu_thu += phu_thu
                tong_calo += calo
                
                ten_tieng_viet_bill = menu_data.get(ten_mon_chot, {}).get("ten_tieng_viet", ten_mon_chot)
                chi_tiet_bill.append({
                    "Ngăn": ngan.replace('Ngan_', 'Ngăn '),
                    "Tên Món": ten_tieng_viet_bill,
                    "Ghi Chú": ghi_chu,
                    "Giá Gốc": f"{gia_goc:,}",
                    "Phụ Thu": f"{phu_thu:,}",
                    "Thành Tiền": f"{thanh_tien_mon:,}"
                })
                
                mon_excel = f"{ten_tieng_viet_bill} ({ghi_chu})" if ghi_chu != "" else ten_tieng_viet_bill
                chi_tiet_luu_excel.append(mon_excel)

        st.markdown("---")
        if len(chi_tiet_bill) > 0:
            st.markdown("### 📋 Chi tiết món ăn")
            st.markdown("""
            <style>
                table { width: 100% !important; font-family: Arial, sans-serif !important; background-color: #ffffff !important; border: 2px solid #333 !important; }
                th { font-size: 22px !important; font-weight: 900 !important; color: #ffffff !important; background-color: #ff4b4b !important; text-align: left !important; border: 1px solid #333 !important; padding: 12px !important; }
                td { font-size: 20px !important; font-weight: 900 !important; color: #000000 !important; background-color: #ffffff !important; border: 1px solid #ddd !important; padding: 10px !important; }
            </style>
            """, unsafe_allow_html=True)
            df_bill = pd.DataFrame(chi_tiet_bill)
            st.table(df_bill)
            tong_truoc_giam_gia = tong_tien_goc + tong_phu_thu
        else:
            st.warning("Chưa có món ăn nào hợp lệ trên khay.")

if img_file is not None and len(chi_tiet_bill) > 0:
    with col1:
        st.markdown("---")
        st.markdown("### 🎟️ Mã giảm giá")
        ma_voucher = st.text_input("Nhập mã Voucher (Nếu có):").strip()
        so_tien_giam = 0
        
        if ma_voucher.lower() == "canteen":
            so_tien_giam = int(tong_truoc_giam_gia * 0.1)
            st.success(f"🎉 Áp dụng mã thành công! Bạn được giảm 10% (-{so_tien_giam:,} VNĐ)")
        elif ma_voucher.lower() == "banmoi":
            if 50000 <= tong_truoc_giam_gia <= 89000:
                so_tien_giam = 5000
                st.success(f"🎉 Áp dụng mã thành công! Bạn được giảm 5,000 VNĐ")
            elif 90000 <= tong_truoc_giam_gia <= 100000:
                so_tien_giam = 10000
                st.success(f"🎉 Áp dụng mã thành công! Bạn được giảm 10,000 VNĐ")
            elif tong_truoc_giam_gia > 100000:
                so_tien_giam = 10000 + ((tong_truoc_giam_gia - 100000) // 20000) * 2000
                st.success(f"🎉 Áp dụng mã thành công! Bạn được giảm {so_tien_giam:,} VNĐ")
            else:
                st.warning("⚠️ Mã 'banmoi' chỉ áp dụng cho hóa đơn từ 50,000 VNĐ trở lên.")
        elif ma_voucher != "":
            st.error("❌ Mã voucher không hợp lệ!")
            
        tong_thuc_thu = tong_truoc_giam_gia - so_tien_giam
        
        st.markdown("### 💳 Tổng Kết Thanh Toán")
        col_kq1, col_kq2 = st.columns(2)
        with col_kq1:
            st.info(f"**🔥 TỔNG CALO:** {tong_calo} kcal")
            st.write(f"Tiền món gốc: **{tong_tien_goc:,} VNĐ**")
            st.write(f"Tiền Topping (Cơm/Trứng): **+{tong_phu_thu:,} VNĐ**")
            st.write(f"Giảm giá Voucher: **-{so_tien_giam:,} VNĐ**")
        with col_kq2:
            st.success(f"**💰 THỰC THU:** {tong_thuc_thu:,} VNĐ")
        
        st.markdown("### 💳 Phương Thức Thanh Toán")
        phuong_thuc = st.radio("Chọn phương thức thanh toán:", ("Chuyển khoản (VietQR)", "Tiền mặt"))
        
        if phuong_thuc == "Chuyển khoản (VietQR)":
            st.markdown("### 📱 Quét mã để thanh toán")
            qr_url = f"https://img.vietqr.io/image/MB-0868350939-compact.png?amount={tong_thuc_thu}&addInfo=ThanhToanCanteen"
            st.image(qr_url, width=250)
        else:
            st.markdown("### 💵 Thanh toán tiền mặt")
            tien_khach_dua = st.number_input("Tiền khách đưa (VNĐ):", min_value=0, value=0, step=1000)
            tien_thoi = tien_khach_dua - tong_thuc_thu
            
            if tien_khach_dua > 0:
                if tien_thoi >= 0:
                    st.success(f"Tiền thối lại: **{tien_thoi:,} VNĐ**")
                else:
                    st.error(f"Khách đưa thiếu: **{abs(tien_thoi):,} VNĐ**")
        
        if st.button("💾 Lưu Hóa Đơn Lịch Sử & Thu Thập Dữ Liệu", use_container_width=True):
            thoi_gian_hien_tai = datetime.now()
            thoi_gian_xuat_bill = thoi_gian_hien_tai.strftime("%Y-%m-%d %H:%M:%S")
            chuoi_mon_an = ", ".join(chi_tiet_luu_excel)
            
            lich_su = {
                "Thời gian": thoi_gian_xuat_bill,
                "Chi tiết món ăn": chuoi_mon_an,
                "Tiền Gốc": tong_tien_goc,
                "Tiền Topping": tong_phu_thu,
                "Mã Voucher": ma_voucher if so_tien_giam > 0 else "Không",
                "Giảm Giá": so_tien_giam,
                "Tổng Thực Thu": tong_thuc_thu,
                "Phương Thức": phuong_thuc,
                "Tổng Calo": tong_calo
            }
            
            file_excel = "lich_su_ban_hang.xlsx"
            df_moi = pd.DataFrame([lich_su])
            
            if os.path.exists(file_excel):
                df_cu = pd.read_excel(file_excel)
                df_hop_nhat = pd.concat([df_cu, df_moi], ignore_index=True)
                df_hop_nhat.to_excel(file_excel, index=False)
            else:
                df_moi.to_excel(file_excel, index=False)
            
            thu_muc_goc = "dataset_thu_thap"
            os.makedirs(thu_muc_goc, exist_ok=True)
            
            for item in st.session_state.ket_qua_ai:
                ngan = item["Ngăn"]
                anh_cat = item["Ảnh Cắt"]
                mon_chot = st.session_state.get(f"select_{ngan}", "-- Bỏ trống (Không có món) --")
                
                if mon_chot != "-- Bỏ trống (Không có món) --" and anh_cat.shape[0] > 0 and anh_cat.shape[1] > 0:
                    thu_muc_mon_an = os.path.join(thu_muc_goc, mon_chot)
                    os.makedirs(thu_muc_mon_an, exist_ok=True)
                    
                    ten_file = f"{mon_chot}_{ngan}_{thoi_gian_hien_tai.strftime('%Y%m%d_%H%M%S')}.jpg"
                    duong_dan_file = os.path.join(thu_muc_mon_an, ten_file)
                    
                    anh_cat_bgr = cv2.cvtColor(anh_cat, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(duong_dan_file, anh_cat_bgr)
            
            st.success("✅ Đã lưu hóa đơn thành công vào file lich_su_ban_hang.xlsx!")
            st.info("💡 Hình ảnh đã được lưu vào thư mục 'dataset_thu_thap' để phục vụ việc dạy AI.")