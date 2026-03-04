import os

# Check if reportlab is installed, if not, try to install it
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    print("reportlab not found. Installing...")
    os.system("pip install reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

def create_policy_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Ultrafast AI Corp: Remote Work Policy")
    
    # Content
    c.setFont("Helvetica", 12)
    text = [
        "Policy Number: HR-2024-001",
        "Effective Date: January 1, 2024",
        "",
        "1. Purpose",
        "The purpose of this policy is to define the guidelines and requirements for remote work",
        "at Ultrafast AI Corp, ensuring productivity and maintaining company culture.",
        "",
        "2. Eligibility",
        "All full-time employees are eligible for remote work, subject to manager approval.",
        "Employees must have a stable internet connection and a dedicated workspace.",
        "",
        "3. Working Hours",
        "Remote employees are expected to be available during core business hours:",
        "9:00 AM to 5:00 PM Pacific Time.",
        "",
        "4. Equipment",
        "The company will provide a NeuralChip X1 enabled laptop and a high-resolution monitor.",
        "Employees are responsible for the safe-keeping of company property.",
        "",
        "5. Security",
        "All work must be performed over the company-approved VPN.",
        "Personal devices are NOT permitted for accessing company data.",
        "",
        "6. Contact",
        "For questions regarding this policy, please contact HR at hr@ultrafast-ai.corp."
    ]
    
    y_position = height - 140
    for line in text:
        c.drawString(100, y_position, line)
        y_position -= 20
        
    c.save()
    print(f"PDF created successfully: {filename}")

if __name__ == "__main__":
    create_policy_pdf("company_policy.pdf")
