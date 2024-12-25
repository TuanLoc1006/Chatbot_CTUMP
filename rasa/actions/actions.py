from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from rasa_sdk.events import FollowupAction
from .constant import Constant
from custom.write_file import write_file
import logging
constant  = Constant()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class actionGoiYHocPhi(Action):
    def name(self):
        return "action_goi_y_hocphi"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action Gợi ý học phí")
        user_message = tracker.latest_message.get('text')
        print("user_ask: " + user_message)
        entities = tracker.latest_message['entities']
        print(f"Entity từ input : {entities}")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        response = "Tôi có thông tin học phí của các ngành:<br>- Y khoa,<br>- Răng hàm mặt,<br>- Y học dự phòng<br>- Y học cổ truyền<br>- Dược<br>- Điều dưỡng<br>- Y tế công cộng<br>- Kỹ thuật xét nghiệm y học<br>- Hộ sinh<br>- Kỹ thuật hình ảnh y học"
        dispatcher.utter_message(text=response)
        return [SlotSet("hocphi_nganh", None)]

class actionHocPhi(Action):
    def name(self):
        return "action_hocphi_nganh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")

        input_nguoidung = tracker.latest_message.get('text')

        print(f"Input người dùng: {input_nguoidung}")
        print("ACTION HOI HOC PHI")
        
        infor_ctdt_hoc_phi = constant.get_hocphi_ctdt()
        
        slot_ten_nganh = tracker.get_slot('hocphi_nganh')
        print(f"Slot tên ngành là : {slot_ten_nganh}")
        
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        
        response = "Bạn vui lòng nhập đầy đủ câu hỏi<br> Ví dụ:<br>- Học phí + tên ngành<br>- Chương trình đào tạo + tên ngành"
        res_default = False
        found = False
        try:
            if 'ten_nganh' in role and 'hoi_hocphi' in role:
                response = f"<br>Học phí của ngành {slot_ten_nganh} : "
                # Chính Quy 
                #Trúng tuyển chính quy
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy']:
                    response += "<br>Hệ chính quy :" 
                    response += f"<br>- Đào tạo trúng tuyển chính quy : " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                    found = True
                    res_default = True
                #đào tạo theo nhu cầu xã hội
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh']:
                    response += "<br>- Đào tạo theo nhu cầu xã hội (theo đặt hàng ) : " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                    found = True
                    res_default = True
                #đào tạo hệ quốc tế tiếng anh
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qte_tieng_anh']:
                    response += "<br>- Đào tạo hệ quốc tế tiếng anh : "+ infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qte_tieng_anh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                    found = True
                    res_default = True
                # Liên thông
                # Liên thông chính quy
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy']:
                    response += "<br>Hệ liên thông : "
                    response += "<br>- Đào tạo liên thông hệ chính quy : " + infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                    found = True
                    res_default = True
                #Liên thông theo nhu cầu xã hội
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh']:
                    response += "<br>- Đào tạo liên thông theo nhu cầu xã hội : " + infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh'][slot_ten_nganh]['hocphi'] + "/tín chỉ"
                    found = True
                    res_default = True
            
                if found == False:
                    response = f"Không tìm thấy ngành bạn yêu cầu."
        except KeyError:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br> Ví dụ:<br>- Học phí + tên ngành<br>- Chương trình đào tạo + tên ngành"
            
        if res_default == False:  
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_hocphi_nganh: ' + tracker.latest_message['text'])  
        
        dispatcher.utter_message(text=response) 

        return [SlotSet("hocphi_nganh", None)]

class actionGoiYCTDT(Action):
    def name(self):
        return "action_goi_y_CTDT"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action Gợi ý ctdt")
        response = "Tôi có thông tin chương trình đào tạo của các ngành:<br>- Y khoa,<br>- Răng hàm mặt,<br>- Y học dự phòng<br>- Y học cổ truyền<br>- Dược<br>- Điều dưỡng<br>- Y tế công cộng<br>- Kỹ thuật xét nghiệm y học<br>- Hộ sinh<br>- Kỹ thuật hình ảnh y học"

        dispatcher.utter_message(text=response)
        
        return []

  
class actionCTDT(Action):
    def name(self):
        return "action_CTDT_nganh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        entities = tracker.latest_message['entities']
        print(f"\n\nEntity từ input : {entities}")
        print("ACTION CTĐT ngành")
        infor_ctdt_hoc_phi = constant.get_hocphi_ctdt()
        role = []
        slot_ten_nganh = tracker.get_slot('CTDT_nganh')
        print(f"Slot tên ngành là : {slot_ten_nganh}")
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        response = "Bạn vui lòng nhập đầy đủ câu hỏi<br> Ví dụ:<br>- Học phí + tên ngành<br>- Chương trình đào tạo + tên ngành"
        found = False
        res_default = False
        try:
            if 'ten_nganh' in role and 'hoi_ctdt' in role:
                response = f"<br>Chương trình đào tạo của {slot_ten_nganh} : "
                print("Chương trình đào tạo của ngành là")
                # Chính Quy 
                #Trúng tuyển chính quy
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy']:
                    print("Chương trình đào tạo chính quy")
                    response += "<br>Hệ chính quy :" 
                    response += f"<br>- Đào tạo trúng tuyển chính quy : " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['chinhquy'][slot_ten_nganh]['ctdt'] 
                    found = True
                    res_default = True
                #đào tạo theo nhu cầu xã hội
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh']:
                    response += "<br>- Đào tạo theo nhu cầu xã hội (theo đặt hàng ) : " + infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['nhu_cau_xh'][slot_ten_nganh]['ctdt'] 
                    found = True
                    res_default = True
                #đào tạo hệ quốc tế tiếng anh
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qte_tieng_anh']:
                    response += "<br>- Đào tạo hệ quốc tế tiếng anh : "+ infor_ctdt_hoc_phi['daihoc']['he_chinhquy']['qte_tieng_anh'][slot_ten_nganh]['ctdt'] 
                    found = True
                    res_default = True
            
                # Liên thông
                # Liên thông chính quy
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy']:
                    response += "<br>Hệ liên thông : "
                    response += "<br>- Đào tạo liên thông hệ chính quy : " + infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_chinhquy'][slot_ten_nganh]['ctdt'] 
                    found = True
                    res_default = True
                #Liên thông theo nhu cầu xã hội
                if slot_ten_nganh in infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh']:
                    response += "<br>- Đào tạo liên thông theo nhu cầu xã hội : " + infor_ctdt_hoc_phi['daihoc']['he_lienthong']['lt_nhu_cau_xh'][slot_ten_nganh]['ctdt'] 
                    found = True
                    res_default = True
            
                if found == False:
                    response = f"Không tìm thấy ngành bạn yêu cầu."
        except KeyError:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br> Ví dụ:<br>- Học phí + tên ngành<br>- Chương trình đào tạo + tên ngành"

        if res_default == False:
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_CTDT_nganh: ' + tracker.latest_message['text'])
        dispatcher.utter_message(text=response)
        
        return [SlotSet("CTDT_nganh", None)]
    

class actionGoiYHoiVeDH_LT_SDH(Action):
    def name(self):
        return "action_goi_y_info_daihoc_lienthong_saudaihoc"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action Gợi ý hỏi về đại học, liên thông, sau đại học")
        entities = tracker.latest_message['entities']
        # print(f"\nnEntity từ input : {entities}")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        
        response = "Tôi có thông tin tuyển sinh, học phí, chương trình đào tạo:<br>- Đại học chính quy<br>- Liên thông<br>- Sau đại học"
      
        dispatcher.utter_message(text=response)
        
        return []

class actionGoiYTuyenSinh(Action):
    def name(self):
        return "action_goi_y_thong_tin_tuyen_sinh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = tracker.latest_message['entities']
        print("action Gợi ý tuyển sinh")
        # print(f"\nnEntity từ input : {entities}")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        if 'info' in role and 'thong_tin_tuyen_sinh' in role or 'thong_tin_tuyen_sinh' in role:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Tuyển sinh đại học<br>- Tuyển sinh liên thông<br>- Tuyển sinh sau đại học"
        else:
            response = f"Bạn vui lòng mô tả thêm thông tin"
        dispatcher.utter_message(text=response)
        
        return []
    
class actionTuyenSinhTungNganh(Action):
    def name(self):
        return "action_thong_tin_tuyen_sinh_tung_nganh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = tracker.latest_message['entities']
        print(f"\nnEntity từ input : {entities}")
        print("action Gợi ý tuyển sinh từng ngành")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        slot_tuyensinh = tracker.get_slot('ten_nganh')
        print(f"Slot tuyển sinh là : {slot_tuyensinh}")

        tuyen_sinh_dh_chinhquy_2024 = constant.get_tuyen_sinh_dh_chinhquy_2024()
        response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Tuyển sinh + tên ngành"
        
        try:
            if 'thong_tin_tuyen_sinh' in role and 'ten_nganh' in role:
                response = f"Thông tin tuyển sinh chính quy ngành {slot_tuyensinh} :<br>"
                nganh_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh']
                found = False
                for nganh, thong_tin in nganh_tuyen_sinh.items():
                    if nganh == slot_tuyensinh:
                        response += (
                            f"+ Mã xét tuyển: {thong_tin['ma_xet_tuyen']},<br>"
                            f"+ Phương thức tuyển: {thong_tin['phuong_thuc_tuyen']},<br>"
                            f"+ Tổ hợp: {thong_tin['to_hop']}<br>"
                            f"+ Xét tuyển đợt 1 (dự kiến): "
                            f"Kết quả thi THPT: {thong_tin['xet_tuyen_dot_1_du_kien']['kqua_thpt']}, "
                            f"Dự bị dân tộc: {thong_tin['xet_tuyen_dot_1_du_kien']['du_bi_dan_toc']}, "
                            f"Hợp đồng: {thong_tin['xet_tuyen_dot_1_du_kien']['hop_dong']}"
                        )
                        found = True
                        break

                if not found:
                    response = "Không tìm thấy tên ngành theo yêu cầu"
 
        except KeyError:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Tuyển sinh + tên ngành"
        dispatcher.utter_message(text=response)
        
        return [SlotSet("ten_nganh", None)]

    
class actionThanhToanHocPhi(Action):
    def name(self):
        return "action_thanh_toan_hoc_phi"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = tracker.latest_message['entities']
        print(f"\nnEntity từ input : {entities}")
        print("action thanh toán học phí")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
     
        
        response = f"Thanh toán học phí online xem chi tiết tại đây: http://www.ctump.edu.vn/DesktopModules/NEWS/DinhKem/7991_HUONG-DAN-NOP-HP-ONLINE_moi.pdf"
       
       
        dispatcher.utter_message(text=response)
        
        return []


class actionDiemChuanTungNganh(Action):
    def name(self):
        return "action_diem_chuan_tung_nganh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = tracker.latest_message['entities']
        print(f"\nnEntity từ input : {entities}")
        print("action điểm chuẩn từng ngành")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        slot_tuyensinh = tracker.get_slot('ten_nganh')
        print(f"Slot tuyển sinh là : {slot_tuyensinh}")

        tuyen_sinh_dh_chinhquy_2024 = constant.get_tuyen_sinh_dh_chinhquy_2024()
       
        response = f"Thông tin điểm chuẩn của ngành {slot_tuyensinh} năm 2024 :<br>"
       
        try:
            if 'diem_tuyen_sinh_dh' in role and 'ten_nganh' in role:
                nganh_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh']
                for nganh, thong_tin in nganh_tuyen_sinh.items():
                    
                    if nganh == slot_tuyensinh:
                        response += f"+ Mã xét tuyển: {thong_tin['ma_xet_tuyen']},<br>+ Điểm chuẩn: {thong_tin['diem_chuan']}<br>  + Phương thức tuyển: {thong_tin['phuong_thuc_tuyen']},<br>  + Tổ hợp: {thong_tin['to_hop']}<br>"   
            if 'diem_tuyen_sinh_dh' in role and 'ten_nganh' not in role:
                nganh_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh']
                response = f"Thông tin điểm chuẩn các ngành tuyển sinh:<br>"
                for nganh, thong_tin in nganh_tuyen_sinh.items():
                    response += f"Tên ngành: {nganh}<br>Mức điểm chuẩn: {tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh'][nganh]['diem_chuan']}<br>"
                    
                        
        except KeyError:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Điểm chuẩn + tên ngành"
        dispatcher.utter_message(text=response)
        
        return [SlotSet("ten_nganh", None)]



class actionChiTieuTungNganh(Action):
    def name(self):
        return "action_chi_tieu_tung_nganh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = tracker.latest_message['entities']
        print(f"\nnEntity từ input : {entities}")
        print("action chỉ tiêu từng ngành")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        slot_tuyensinh = tracker.get_slot('ten_nganh')
        print(f"Slot tuyển sinh là : {slot_tuyensinh}")

        tuyen_sinh_dh_chinhquy_2024 = constant.get_tuyen_sinh_dh_chinhquy_2024()
       
        response = f"Thông tin chỉ tiêu của ngành {slot_tuyensinh} năm 2024 :<br>"
       
        try:
            if 'chi_tieu_dh' in role and 'ten_nganh' in role:
                nganh_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh']
                for nganh, thong_tin in nganh_tuyen_sinh.items():
                    # print(type(nganh), type(slot_tuyensinh))
                    if nganh == slot_tuyensinh:
                        response += (
                            # f"+ Mã xét tuyển: {thong_tin['ma_xet_tuyen']},<br>"
                            # f"+ Phương thức tuyển: {thong_tin['phuong_thuc_tuyen']},<br>"
                            # f"+ Tổ hợp: {thong_tin['to_hop']}<br>"
                            f"+ Xét tuyển đợt 1 (dự kiến): "
                            f"Chỉ tiêu kết quả thi THPT: {thong_tin['xet_tuyen_dot_1_du_kien']['kqua_thpt']}, "
                            f"chỉ tiêu dự bị dân tộc: {thong_tin['xet_tuyen_dot_1_du_kien']['du_bi_dan_toc']}, "
                            f"Chỉ tiêu hợp đồng: {thong_tin['xet_tuyen_dot_1_du_kien']['hop_dong']}"
                        )
            if 'chi_tieu_dh' in role and 'ten_nganh' not in role:
                nganh_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh']
                response = f"Thông tin chỉ tiêu các ngành tuyển sinh:<br>"
                for nganh, thong_tin in nganh_tuyen_sinh.items():
                    response += f"Tên ngành: {nganh}<br>- Chỉ tiêu kết quả THPT: {tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh'][nganh]['xet_tuyen_dot_1_du_kien']['kqua_thpt']}<br>- Chỉ tiêu dự bị dân tộc: {tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh'][nganh]['xet_tuyen_dot_1_du_kien']['du_bi_dan_toc']}<br>- Chỉ tiêu hợp đồng: {tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['cac_nganh_tuyen_sinh'][nganh]['xet_tuyen_dot_1_du_kien']['hop_dong']}<br>"              
                    
              
        except KeyError:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Chỉ tiêu + tên ngành"
        dispatcher.utter_message(text=response)
        
        return [SlotSet("ten_nganh", None)]


class actionGoiYTuyenSinhDaiHoc(Action):
    def name(self):
        return "action_goi_y_tuyen_sinh_dai_hoc"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = tracker.latest_message['entities']
        print(f"\nnEntity từ input : {entities}")
        print("ACTION goi ý tuyển sinh đại học, liên thông, sau đại học")

        tuyen_sinh_dh_chinhquy_2024 = constant.get_tuyen_sinh_dh_chinhquy_2024()

        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        slot_tuyensinh = tracker.get_slot('tuyen_sinh_daihoc_chinhquy')
        print(f"Slot tuyển sinh là : {slot_tuyensinh}")
        response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Tuyển sinh đại học<br>- Tuyển sinh liên thông<br>- Tuyển sinh sau đại học"
        try:
            if 'thong_tin_tuyen_sinh' in role and 'dai_hoc_chinh_quy' in role and 'quy_trinh_tuyen_sinh' not in role:
                response = f"<br>Thông tin tuyển sinh {slot_tuyensinh} gôm:<br>"
                if slot_tuyensinh in tuyen_sinh_dh_chinhquy_2024 and slot_tuyensinh == 'đại học chính quy':
                    cac_nganh_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['cac_nganh_tuyen_sinh']
                    thong_tin_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['thong_tin_tuyen_sinh']
                    for nganh, thong_tin in cac_nganh_tuyen_sinh.items():
                        response += f"- {nganh}<br>  + Mã xét tuyển: {thong_tin['ma_xet_tuyen']},<br>  + Phương thức tuyển: {thong_tin['phuong_thuc_tuyen']},<br>  + Tổ hợp: {thong_tin['to_hop']}<br>"
                    response += "<br>" + thong_tin_tuyen_sinh


            if 'thong_tin_tuyen_sinh' in role and 'lien_thong' in role and 'quy_trinh_tuyen_sinh' not in role:
                response = f"<br>Thông tin tuyển sinh {slot_tuyensinh} gôm:<br>"
                if slot_tuyensinh in tuyen_sinh_dh_chinhquy_2024 and slot_tuyensinh == 'liên thông chính quy':
                    trung_cap_len_dai_hoc = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['trung_cap_len_dai_hoc']['cac_nganh_tuyen_sinh']
                    cao_dang_len_dai_hoc = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['cao_dang_len_dai_hoc']['cac_nganh_tuyen_sinh']
                    thong_tin_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['thong_tin_tuyen_sinh']
                    response += "***Trung cấp lên đại học***<br>" 
                    for nganh, thong_tin in trung_cap_len_dai_hoc.items():
                        response += f"- {nganh}<br>  + Mã ngành: {thong_tin['ma_nganh']}<br>  + Mã xét tuyển: {thong_tin['ma_xet_tuyen']},<br>  + Phương thức tuyển: {thong_tin['phuong_thuc_tuyen']}<br>"
                    response += "***Cao đẳng lên đại học***<br>" 
                    for nganh, thong_tin in cao_dang_len_dai_hoc.items():
                        response += f"- {nganh}<br>  + Mã ngành: {thong_tin['ma_nganh']}<br>  + Mã xét tuyển: {thong_tin['ma_xet_tuyen']},<br>  + Phương thức tuyển: {thong_tin['phuong_thuc_tuyen']}<br>"
                    response += "<br>" + thong_tin_tuyen_sinh
                    
            if 'thong_tin_tuyen_sinh' in role and 'sau_dai_hoc' in role and 'quy_trinh_tuyen_sinh' not in role:
                response = f"<br>Thông tin tuyển sinh {slot_tuyensinh} gôm:<br>"
                response += tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]
        except KeyError:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Tuyển sinh đại học<br>- Tuyển sinh liên thông<br>- Tuyển sinh sau đại học"
        if response == "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Tuyển sinh đại học<br>- Tuyển sinh liên thông<br>- Tuyển sinh sau đại học":
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_goi_y_tuyen_sinh_dai_hoc: ' + tracker.latest_message['text'])
            
        dispatcher.utter_message(text=response)
        
        return  [SlotSet("tuyen_sinh_daihoc_chinhquy", None)]
    

class actionGoiYQuyTrinhTuyenSinh(Action):
    def name(self):
        return "action_goi_y_quy_trinh_tuyen_sinh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action Gợi ý các quy trinh")
        entities = tracker.latest_message['entities']
        # print(f"\nnEntity từ input : {entities}")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        
        response = f"Tôi có thông tin quy trình tuyển sinh đại học, liên thông, sau đại học nè<br>Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- quy trình tuyển sinh đại học<br>- quy trình tuyển sinh liên thông<br>- quy trình tuyển sinh sau đại học"
        dispatcher.utter_message(text=response)
        
        return []
    

class actionTuyenSinhDaiHocLienThongSauDaiHoc(Action):
    def name(self):
        return "action_quy_trinh_tuyen_sinh_dh_lt_sdh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("ACTION tuyển sinh đại học, liên thông, sau đại học")
        entities = tracker.latest_message['entities']
        print(f"\nnEntity từ input : {entities}")

        tuyen_sinh_dh_chinhquy_2024 = constant.get_tuyen_sinh_dh_chinhquy_2024()
        # print(tuyen_sinh_dh_chinhquy_2024)
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        slot_tuyensinh = tracker.get_slot('tuyen_sinh_daihoc_chinhquy')
        print(f"Slot tuyển sinh là : {slot_tuyensinh}")
        response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Quy trình tuyển sinh đại học<br>- Quy trình tuyển sinh liên thông<br>- Quy trình tuyển sinh sau đại học."
        try:
            if  'quy_trinh_tuyen_sinh' in role and 'thong_tin_tuyen_sinh' in role and 'dai_hoc_chinh_quy' in role:
                response = f"<br>Quy trình tuyển sinh {slot_tuyensinh} gôm:<br>"
                phuong_thuc_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['phuong_thuc_hinh_thuc_tuyen_sinh']['phuong_thuc']
                hinh_thuc_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['phuong_thuc_hinh_thuc_tuyen_sinh']['phuong_thuc']
            
                response += "- Phương thức: "+phuong_thuc_tuyen_sinh
                response += "<br>- Hình thức: "+hinh_thuc_tuyen_sinh

            if  'quy_trinh_tuyen_sinh' in role and 'thong_tin_tuyen_sinh' in role and 'lien_thong' in role:
                response = f"<br>Quy trình tuyển sinh {slot_tuyensinh} gôm:<br>"
                hinh_thuc_tuyen_sinh = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['hinh_thuc_tuyen_sinh']['thong_tin']
                lt_chinh_quy_trung_cap_len_dh_Y_khoa = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['hinh_thuc_tuyen_sinh']['lt_chinh_quy_trung_cap_len_dh_Y_khoa']
                lt_chinh_quy_trung_cap_len_dh_yhct_d_yhdp_dd_xnyh = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['hinh_thuc_tuyen_sinh']['lt_chinh_quy_trung_cap_len_dh_yhct_d_yhdp_dd_xnyh']
                lt_chinh_quy_cao_dang_len_dh_d_dd_xnyh = tuyen_sinh_dh_chinhquy_2024[slot_tuyensinh]['hinh_thuc_tuyen_sinh']['lt_chinh_quy_cao_dang_len_dh_d_dd_xnyh']
                response += f""+hinh_thuc_tuyen_sinh
                response += f"<br>"+lt_chinh_quy_trung_cap_len_dh_Y_khoa
                response += f"<br>"+lt_chinh_quy_trung_cap_len_dh_yhct_d_yhdp_dd_xnyh
                response += f"<br>"+lt_chinh_quy_cao_dang_len_dh_d_dd_xnyh
            if  'quy_trinh_tuyen_sinh' in role and 'thong_tin_tuyen_sinh' in role and 'sau_dai_hoc' in role:
                response = f"<br>Quy trình tuyển sinh {slot_tuyensinh} gôm:<br>"
                response += f"Sau đại học có nhiều thông tin nên bạn xem chi tiết ở đây nha: http://www.ctump.edu.vn/Default.aspx?tabid=1036"
        except KeyError:
               response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Quy trình tuyển sinh đại học<br>- Quy trình tuyển sinh liên thông<br>- Quy trình tuyển sinh sau đại học."
        if response == "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Quy trình tuyển sinh đại học<br>- Quy trình tuyển sinh liên thông<br>- Quy trình tuyển sinh sau đại học.":
                file_writer = write_file()
                file_writer.get_ghi_log_file('action_quy_trinh_tuyen_sinh_dh_lt_sdh: ' + tracker.latest_message['text'])
        dispatcher.utter_message(text=response)
        
        return  [SlotSet("tuyen_sinh_daihoc_chinhquy", None)]


class actionGoiYDoiTuong(Action):
    def name(self):
        return "action_goi_y_doi_tuong"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("\naction Gợi ý dối tượng")
        entities = tracker.latest_message['entities']
        print(f"nEntity từ input : {entities}")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        
        response = f"Tôi có thông tin về đối tượng tuyển sinh đại học, liên thông, sau đại học"
        dispatcher.utter_message(text=response)
        
        return []
    

class actionDoiTuongTuyenSinh(Action):
    def name(self):
        return "action_doi_tuong_tuyen_sinh_dh_lt_sdh"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = tracker.latest_message['entities']
        print(f"\nnEntity từ input : {entities}")
        print("ACTION đối tượng tuyển sinh đại học, liên thông, sau đại học")

        tuyen_sinh_dh_chinhquy_2024 = constant.get_tuyen_sinh_dh_chinhquy_2024()
        response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Đối tượng tuyển sinh đại học<br>- Đối tượng tuyển sinh liên thông<br>- Đối tượng tuyển sinh sau đại học."
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        # slot_tuyensinh = tracker.get_slot('tuyen_sinh_daihoc_chinhquy')
        # print(f"Slot tuyển sinh là : {slot_tuyensinh}")
        # response = "đói tượng tuyển."
        try:
            if  'doi_tuong_tuyen_sinh' in role and 'thong_tin_tuyen_sinh' in role and ('dai_hoc_chinh_quy' not in role or 'lien_thong' not in role or 'sau_dai_hoc' not in role):
                response = f"<br>- Đối tượng tuyển sinh đại học gôm:<br>"
                doi_tuong_tuyen_sinh_dh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['doi_tuong_tuyen_sinh']
                vung_tuyen_sinh_dh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['vung_tuyen_sinh']
                response += "+ Đối tượng: "+doi_tuong_tuyen_sinh_dh
                response += "<br>+ Vùng tuyển sinh: "+vung_tuyen_sinh_dh
                
                response += f"<br>- Đối tượng tuyển sinh liên thông gôm:<br>"
                doi_tuong_tuyen_sinh_lt = tuyen_sinh_dh_chinhquy_2024['liên thông chính quy']['doi_tuong_tuyen_sinh']
                tieu_chuan_tuyen_sinh_lt_chinh_tri = tuyen_sinh_dh_chinhquy_2024['liên thông chính quy']['tieu_chuan_tuyen_sinh']['chinh_tri']
                doi_tuong_tuyen_sinh_lt_van_hoa_chuyen_mon = tuyen_sinh_dh_chinhquy_2024['liên thông chính quy']['tieu_chuan_tuyen_sinh']['van_hoa_chuyen_mon']
                tieu_chuan_tuyen_sinh_lt_suc_khoe = tuyen_sinh_dh_chinhquy_2024['liên thông chính quy']['tieu_chuan_tuyen_sinh']['suc_khoe']
                response += "+ Đối tượng: "+doi_tuong_tuyen_sinh_lt
                response += "<br>+ Tiêu chuẩn chính trị: "+tieu_chuan_tuyen_sinh_lt_chinh_tri
                response += "<br>+ Tiêu chuẩn văn hóa chuyên môn: "+doi_tuong_tuyen_sinh_lt_van_hoa_chuyen_mon
                response += "<br>+ Tiêu chuẩn sức khỏe: "+tieu_chuan_tuyen_sinh_lt_suc_khoe
                
                doi_tuong_tuyen_sinh_sdh = tuyen_sinh_dh_chinhquy_2024['sau đại học chính quy']
                response += f"<br>- Đối tượng tuyển sinh sau đại học gôm:<br>"+doi_tuong_tuyen_sinh_sdh

            if  'doi_tuong_tuyen_sinh' in role and 'thong_tin_tuyen_sinh' in role and 'dai_hoc_chinh_quy' in role:
                response = f"<br>- Đối tượng tuyển sinh đại học gồm <br>"
                doi_tuong_tuyen_sinh_dh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['doi_tuong_tuyen_sinh']
                vung_tuyen_sinh_dh = tuyen_sinh_dh_chinhquy_2024['đại học chính quy']['vung_tuyen_sinh']
                response += "+ Đối tượng: "+doi_tuong_tuyen_sinh_dh
                response += "<br>+ Vùng tuyển sinh: "+vung_tuyen_sinh_dh
            if  'doi_tuong_tuyen_sinh' in role and 'thong_tin_tuyen_sinh' in role and 'lien_thong' in role:
                response = f"<br>- Đối tượng tuyển sinh liên thông gồm:<br>"
                doi_tuong_tuyen_sinh_lt = tuyen_sinh_dh_chinhquy_2024['liên thông chính quy']['doi_tuong_tuyen_sinh']
                tieu_chuan_tuyen_sinh_lt_chinh_tri = tuyen_sinh_dh_chinhquy_2024['liên thông chính quy']['tieu_chuan_tuyen_sinh']['chinh_tri']
                doi_tuong_tuyen_sinh_lt_van_hoa_chuyen_mon = tuyen_sinh_dh_chinhquy_2024['liên thông chính quy']['tieu_chuan_tuyen_sinh']['van_hoa_chuyen_mon']
                tieu_chuan_tuyen_sinh_lt_suc_khoe = tuyen_sinh_dh_chinhquy_2024['liên thông chính quy']['tieu_chuan_tuyen_sinh']['suc_khoe']
                response += "+ Đối tượng: "+doi_tuong_tuyen_sinh_lt
                response += "<br>+ Tiêu chuẩn chính trị: "+tieu_chuan_tuyen_sinh_lt_chinh_tri
                response += "<br>+ Tiêu chuẩn văn hóa chuyên môn: "+doi_tuong_tuyen_sinh_lt_van_hoa_chuyen_mon
                response += "<br>+ Tiêu chuẩn sức khỏe: "+tieu_chuan_tuyen_sinh_lt_suc_khoe

            if  'doi_tuong_tuyen_sinh' in role and 'thong_tin_tuyen_sinh' in role and 'sau_dai_hoc' in role:
                response = f"<br>- Đối tượng tuyển sinh sau đại học gôm:<br>"+doi_tuong_tuyen_sinh_sdh
        except KeyError:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Đối tượng tuyển sinh đại học<br>- Đối tượng tuyển sinh liên thông<br>- Đối tượng tuyển sinh sau đại học."
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_doi_tuong_tuyen_sinh_dh_lt_sdh: ' + tracker.latest_message['text'])
        dispatcher.utter_message(text=response)
        
        return  []


class actionThongTinKyTucXa(Action):
    def name(self):
        return "action_thong_tin_ky_tuc_xa"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action thông tin ký túc xá")
        entities = tracker.latest_message['entities']
        # print(f"\nnEntity từ input : {entities}")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
        
        response = f"Tôi có thông tin về ký túc xá xem chi tiết tại đây: https://www.facebook.com/KytucxaTDHYDCT/"
        dispatcher.utter_message(text=response)
        
        return []

class actionDiadiemkhoa(Action):
    def name(self):
        return "action_diadiem_khoa"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
       
        list_thong_tin = constant.get_diadiem_khoa()
       
        #lấy ra entity từ câu chat
        entities = tracker.latest_message['entities']
        print(f"\n\nEntities từ input: {entities} ")
       
        response = "Vui lòng nhập đầy đủ câu hỏi: Địa điểm + tên khoa cần tìm."
        slot_tenkhoa = tracker.get_slot('ten_khoa')
        print(f"slot tên khoa khi hỏi địa điểm là {slot_tenkhoa}")
        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)
                               
        if 'where' in role and 'ten_khoa' in role:
            if slot_tenkhoa in list_thong_tin:
                response = f"Địa chỉ của {list_thong_tin[slot_tenkhoa]['Tên']:} : \n"  
                response += list_thong_tin[slot_tenkhoa]['Địa chỉ']
            else : response = "Tìm khong thấy tên khoa"
        dispatcher.utter_message(text=response)
       
        return [SlotSet("ask", None) , SlotSet("ten_khoa", None)]



class ActionBanGiamHieu(Action):
    def name(self):
        return "action_thong_tin_hieu_ban_giam_hieu"
   
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Tạo dictionary ánh xạ các role với thông điệp tương ứng
        info_ban_giam_hieu = constant.get_info_ban_giam_hieu()
            
        entities = tracker.latest_message['entities']
        print(f"action thông tin ban giám hiệu")
        print(f"Entities từ input: {entities}")
        response = "Xin lỗi tôi không có thông tin về câu hỏi này! Vui lòng cung cấp chi tiết thông tin."

        role = []
        for entity in entities:
            if entity.get('role') not in role:
                role.append(entity.get('role'))
        print("mảng role: ", role)

        try:
            if 'hieu_truong' in role and 'pho_hieu_truong' not in role:
                response = f"Thông tin hiệu trưởng trường đại học Y Dược Cần thơ:"
                response += f"{info_ban_giam_hieu['hieu_truong']}"
            elif 'pho_hieu_truong' in role and 'hieu_truong' not in role:
                response = f"Thông tin phó hiệu trưởng trường đại học Y Dược Cần thơ:"
                response += f"{info_ban_giam_hieu['pho_hieu_truong']}"
            else:
                response = f"Thông tin hiệu trưởng và phó hiệu trưởng trường đại học Y Dược Cần thơ:"
                response += f"{info_ban_giam_hieu['thong_tin']}"
              
        except KeyError:
            response = "Bạn vui lòng nhập đầy đủ câu hỏi<br>Ví dụ:<br>- Đối tượng tuyển sinh đại học<br>- Đối tượng tuyển sinh liên thông<br>- Đối tượng tuyển sinh sau đại học."
            file_writer = write_file()
            file_writer.get_ghi_log_file('action_doi_tuong_tuyen_sinh_dh_lt_sdh: ' + tracker.latest_message['text'])
        dispatcher.utter_message(text=response)
        
        return  []
           


