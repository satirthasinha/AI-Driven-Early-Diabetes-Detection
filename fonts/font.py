from fpdf import FPDF

pdf = FPDF()
pdf.add_page()

# Add fonts (ensure TTF files exist in fonts/ folder)
pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
pdf.add_font('NotoBengali', '', 'fonts/bengali.ttf', uni=True)
pdf.add_font('NotoHindi', '', 'fonts/hindi.ttf', uni=True)

# English text
pdf.set_font('DejaVu', '', 14)
pdf.cell(0, 10, 'Hello World (English)', ln=1)

# Bengali text
pdf.set_font('NotoBengali', '', 14)
pdf.cell(0, 10, 'বাংলা লেখা', ln=1)

# Hindi text
pdf.set_font('NotoHindi', '', 14)
pdf.cell(0, 10, 'हिन्दी लिखावट', ln=1)

pdf.output('multi_font_example.pdf')
