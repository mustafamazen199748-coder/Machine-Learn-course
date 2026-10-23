# استخدام صورة بايثون 3.13 الخفيفة
FROM python:3.13.5-slim-bookworm

# نسخ أدوات uv السريعة من صورتها الرسمية مباشرة
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# تحديد مجلد العمل داخل الحاوية
WORKDIR /code

# تفعيل مسار البيئة الافتراضية
ENV PATH="/code/.venv/bin:$PATH"

# نسخ ملفات التبعيات لتثبيتها أولاً
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# تثبيت الحزم بدقة وتجاوز تثبيت مجلد المشروع كحزمة
RUN uv sync --locked --no-install-project

# نسخ ملف الخادم وملف النموذج المدرب
COPY "predict.py" "model.bin" ./

# فتح المنفذ 9696 لاستقبال الاتصالات
EXPOSE 9696

# أمر تشغيل السيرفر كنقطة انطلاق رئيسية
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]