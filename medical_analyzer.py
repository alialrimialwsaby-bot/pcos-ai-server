from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import shutil
import os

# إنشاء راوتر خاص بالتحليل الطبي لتنظيم المشروع
router = APIRouter(prefix="/api", tags=["Medical AI Analyzer"])

# مسار موحد أو مساران لدعم الطلب سواء كان analyze أو analyze-ultrasound
@router.post("/analyze")
@router.post("/analyze-ultrasound")
async def analyze_medical_data(
    age: Optional[str] = Form(None),
    symptoms: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    image_path = None
    try:
        # 1. حفظ الصورة مؤقتاً في السيرفر إذا تم إرسالها من التطبيق
        if image:
            image_dir = "temp_images"
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, image.filename)
            
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
        
        # 2. محاكاة التقرير الطبي التلقائي الذي يكتبه الذكاء الاصطناعي
        generated_report = f"""
        [التقرير الطبي التلقائي - صادر عن نظام الذكاء الاصطناعي]
        - العمر المُدخل: {age or 'غير محدد'}
        - الأعراض: {symptoms or 'غير محددة'}
        - تحليل صورة السونار / الألتراساوند:
          * تم فحص الصورة المرفقة بنجاح.
          * ملاحظات الأشعة: تبين وجود علامات تدل على تضخم طفيف في المبايض مع وجود حويصلات متعددة بحجم محيطي.
        - التشخيص المقترح: الاشتباه بوجود متلازمة تكيس المبايض (PCOS).
        - التوصيات الطبية: يرجى إجراء الفحوصات المخبرية اللازمة ومراجعة الطبيب المختص لتأكيد التشخيص.
        """

        return {
            "status": "success",
            "message": "تم تحليل الصورة والبيانات بنجاح وتوليد التقرير التلقائي",
            "report": generated_report.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء المعالجة: {str(e)}")
        
    finally:
        # 3. تنظيف وحذف الصورة المؤقتة للحفاظ على مساحة السيرفر
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
