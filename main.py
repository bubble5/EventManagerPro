import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import qrcode
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import uuid
import csv
import pandas as pd
from datetime import datetime
import cv2
import threading
import sys
import webbrowser
import logging
import subprocess
import numpy as np
import sys
import os

def get_resource_path(relative_path):
    """Get absolute path to resource for both dev and PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='event_manager.log',
    filemode='a'
)

# Try to import pyzbar, but provide fallback if not available
try:
    from pyzbar.pyzbar import decode
    ZBAR_AVAILABLE = True
    logging.info("pyzbar library loaded successfully")
except ImportError as e:
    logging.warning(f"pyzbar import failed: {e}. Using OpenCV fallback.")
    ZBAR_AVAILABLE = False
except Exception as e:
    logging.error(f"ZBar library error: {e}")
    ZBAR_AVAILABLE = False

# Fallback QR decoder using OpenCV
class SimpleQRDecoder:
    def __init__(self):
        try:
            self.qr_decoder = cv2.QRCodeDetector()
            self.available = True
            logging.info("OpenCV QR Code Detector initialized")
        except Exception as e:
            logging.error(f"Failed to initialize OpenCV QR decoder: {e}")
            self.available = False
    
    def decode(self, image):
        try:
            if not self.available:
                return []
            
            # Convert PIL image to OpenCV format if needed
            if hasattr(image, 'tobytes'):
                if isinstance(image, np.ndarray):
                    # Already a numpy array
                    img_array = image
                else:
                    # Convert PIL to numpy
                    img_array = np.array(image)
                
                # Convert to BGR if needed
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif img_array.shape[2] == 3:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                # Assume it's already an OpenCV image
                img_array = image
            
            # Decode QR code
            data, bbox, _ = self.qr_decoder.detectAndDecode(img_array)
            
            if data and bbox is not None:
                # Create a mock object similar to pyzbar's result
                class MockDecodedObject:
                    def __init__(self, data):
                        self.data = data.encode('utf-8') if isinstance(data, str) else data
                
                return [MockDecodedObject(data)]
            return []
            
        except Exception as e:
            logging.error(f"QR decoding failed: {e}")
            return []

# Initialize the appropriate decoder
if ZBAR_AVAILABLE:
    QRDecoder = decode
    logging.info("Using pyzbar for QR decoding")
else:
    simple_decoder = SimpleQRDecoder()
    if simple_decoder.available:
        QRDecoder = simple_decoder.decode
        logging.info("Using OpenCV fallback for QR decoding")
    else:
        QRDecoder = lambda x: []
        logging.warning("No QR decoder available - scanning will not work")

def check_packages():
    """Check if required packages are installed (non-blocking version)"""
    print("="*60)
    print("EVENT MANAGER PRO - PACKAGE CHECK")
    print("="*60)
    
    required_packages = [
        ('Pillow', 'PIL'),
        ('qrcode', 'qrcode'),
        ('opencv-python', 'cv2'),
        ('pandas', 'pandas'),
    ]
    
    optional_packages = [
        ('pyzbar', 'pyzbar'),
    ]
    
    print("\nChecking required packages...")
    missing_required = []
    for package, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_required.append(package)
    
    print("\nChecking optional packages...")
    for package, import_name in optional_packages:
        try:
            __import__(import_name)
            print(f"✓ {package} is installed (recommended)")
        except ImportError:
            print(f"✗ {package} is NOT installed (optional)")
    
    if missing_required:
        print(f"\n⚠️  WARNING: Missing required packages: {', '.join(missing_required)}")
        print("\nSome features may not work properly.")
        print("To install missing packages, run:")
        for package in missing_required:
            print(f"  pip install {package}")
        print("\nThe application will continue, but may have limited functionality.")
    else:
        print("\n✅ All required packages are installed!")
    
    print("\n" + "="*60)
    print("Starting Event Manager Pro...")
    print("="*60 + "\n")
    
    return len(missing_required) == 0

# ==================== CORE CLASSES ====================

class InvitationCardGenerator:
    # In InvitationCardGenerator.__init__:
    def __init__(self, output_dir=None):
        if output_dir is None:
            # Use executable directory
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(base_dir, "invitation_cards")
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.guest_ids = {}
        logging.info(f"InvitationCardGenerator initialized with output dir: {self.output_dir}")
        
    def generate_guest_id(self, guest_name):
        """Generate unique guest ID"""
        guest_id = str(uuid.uuid4())[:8].upper()
        self.guest_ids[guest_name] = guest_id
        return guest_id
    
    def create_qr_code(self, guest_data, filename):
        """Generate QR code with guest information and ID"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        
        # Create structured QR content for easy parsing
        guest_id = self.generate_guest_id(guest_data['name'])
        qr_content = f"""
        GUEST_ID: {guest_id}
        NAME: {guest_data['name']}
        EVENT: {guest_data['event_name']}
        DATE: {guest_data['event_date']}
        TIME: {guest_data['event_time']}
        VENUE: {guest_data['venue']}
        RSVP: {guest_data['rsvp_date']}
        ADDITIONAL_INFO: {guest_data.get('additional_info', '')}
        """
        
        qr.add_data(qr_content)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(filename)
        
        # Save to master guest list
        self.save_to_master_list(guest_data, guest_id)
        
        logging.info(f"QR code created for {guest_data['name']} with ID: {guest_id}")
        return qr_img, guest_id
    
    def save_to_master_list(self, guest_data, guest_id):
        """Save guest to master list for validation"""
        master_file = "guest_master_list.csv"
        file_exists = os.path.exists(master_file)
        
        with open(master_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow([
                    'guest_id', 'name', 'email', 'event_name', 'event_date',
                    'event_time', 'venue', 'rsvp_date', 'additional_info',
                    'invitation_created', 'status'
                ])
            
            writer.writerow([
                guest_id,
                guest_data['name'],
                guest_data.get('email', ''),
                guest_data['event_name'],
                guest_data['event_date'],
                guest_data['event_time'],
                guest_data['venue'],
                guest_data['rsvp_date'],
                guest_data.get('additional_info', ''),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Invited'
            ])
        
        logging.debug(f"Guest {guest_data['name']} saved to master list")
    
    def create_invitation_card(self, guest_data):
        """Create a beautiful invitation card with QR code"""
        try:
            # Card dimensions
            width, height = 800, 650
            
            # Create background image
            background_color = (240, 240, 255)  # Light purple
            card = Image.new('RGB', (width, height), background_color)
            draw = ImageDraw.Draw(card)
            
            # Try to load fonts
            try:
                title_font = ImageFont.truetype("arial.ttf", 36)
                header_font = ImageFont.truetype("arial.ttf", 24)
                text_font = ImageFont.truetype("arial.ttf", 18)
                small_font = ImageFont.truetype("arial.ttf", 14)
            except:
                # Fallback to default fonts
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add decorative border
            border_color = (75, 0, 130)  # Purple
            draw.rectangle([10, 10, width-10, height-10], outline=border_color, width=3)
            
            # Add title
            title = "You're Invited!"
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            draw.text(((width - title_width) // 2, 50), title, fill=border_color, font=title_font)
            
            # Add guest name
            guest_text = f"Dear {guest_data['name']},"
            guest_bbox = draw.textbbox((0, 0), guest_text, font=header_font)
            guest_width = guest_bbox[2] - guest_bbox[0]
            draw.text(((width - guest_width) // 2, 120), guest_text, fill=(0, 0, 0), font=header_font)
            
            # Add event details
            y_position = 180
            details = [
                f"Event: {guest_data['event_name']}",
                f"Date: {guest_data['event_date']}",
                f"Time: {guest_data['event_time']}",
                f"Venue: {guest_data['venue']}",
                f"RSVP by: {guest_data['rsvp_date']}"
            ]
            
            for detail in details:
                detail_bbox = draw.textbbox((0, 0), detail, font=text_font)
                detail_width = detail_bbox[2] - detail_bbox[0]
                draw.text(((width - detail_width) // 2, y_position), detail, fill=(0, 0, 0), font=text_font)
                y_position += 40
            
            # Generate and add QR code
            qr_filename = os.path.join(self.output_dir, f"qr_{guest_data['name'].replace(' ', '_')}.png")
            qr_img, guest_id = self.create_qr_code(guest_data, qr_filename)
            
            # Resize QR code
            qr_size = 150
            qr_img = qr_img.resize((qr_size, qr_size))
            
            # Paste QR code on card
            qr_x = (width - qr_size) // 2
            qr_y = y_position + 20
            card.paste(qr_img, (qr_x, qr_y))
            
            # Add Guest ID
            guest_id_text = f"Guest ID: {guest_id}"
            guest_id_bbox = draw.textbbox((0, 0), guest_id_text, font=text_font)
            guest_id_width = guest_id_bbox[2] - guest_id_bbox[0]
            draw.text(((width - guest_id_width) // 2, qr_y - 30), guest_id_text, fill=(75, 0, 130), font=text_font)
            
            # Add QR code label
            qr_label = "Scan for details"
            qr_label_bbox = draw.textbbox((0, 0), qr_label, font=small_font)
            qr_label_width = qr_label_bbox[2] - qr_label_bbox[0]
            draw.text(((width - qr_label_width) // 2, qr_y + qr_size + 10), qr_label, fill=(100, 100, 100), font=small_font)
            
            # Add additional info if available
            if 'additional_info' in guest_data and guest_data['additional_info']:
                info_text = guest_data['additional_info']
                if len(info_text) > 50:
                    words = info_text.split()
                    lines = []
                    current_line = []
                    for word in words:
                        if len(' '.join(current_line + [word])) <= 50:
                            current_line.append(word)
                        else:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                    if current_line:
                        lines.append(' '.join(current_line))
                    info_text = '\n'.join(lines)
                
                info_lines = info_text.split('\n')
                for i, line in enumerate(info_lines):
                    line_bbox = draw.textbbox((0, 0), line, font=small_font)
                    line_width = line_bbox[2] - line_bbox[0]
                    draw.text(((width - line_width) // 2, qr_y + qr_size + 40 + (i * 20)), line, fill=(50, 50, 50), font=small_font)
            
            # Save the invitation card
            card_filename = os.path.join(self.output_dir, f"invitation_{guest_data['name'].replace(' ', '_')}.png")
            card.save(card_filename)
            
            logging.info(f"Invitation card created: {card_filename}")
            return card_filename, guest_id
            
        except Exception as e:
            logging.error(f"Failed to create invitation card: {e}")
            raise
    
    def send_invitation_email(self, guest_data, card_filename, guest_id, email_config):
        """Send invitation via email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = guest_data['email']
            msg['Subject'] = f"Invitation: {guest_data['event_name']}"
            
            # Email body
            body = f"""
            <html>
            <body>
                <h2 style="color: #4B0082;">You're Invited!</h2>
                <p>Dear <strong>{guest_data['name']}</strong>,</p>
                
                <p>We are delighted to invite you to:</p>
                
                <div style="background-color: #f0f0ff; padding: 15px; border-left: 4px solid #4B0082;">
                    <h3>{guest_data['event_name']}</h3>
                    <p><strong>Date:</strong> {guest_data['event_date']}</p>
                    <p><strong>Time:</strong> {guest_data['event_time']}</p>
                    <p><strong>Venue:</strong> {guest_data['venue']}</p>
                    <p><strong>Your Guest ID:</strong> <code style="background: #f0f0f0; padding: 2px 5px; border-radius: 3px;">{guest_id}</code></p>
                </div>
                
                <p>Please RSVP by: <strong>{guest_data['rsvp_date']}</strong></p>
                
                {f"<p><em>{guest_data.get('additional_info', '')}</em></p>" if guest_data.get('additional_info') else ""}
                
                <p>Your invitation card with QR code is attached. Please present it at the event.</p>
                
                <p>We look forward to seeing you!</p>
                
                <hr>
                <p style="color: #666; font-size: 12px;">
                    This is an automated invitation. Please do not reply to this email.
                </p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Attach invitation card
            with open(card_filename, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', 
                             filename=f"Invitation_{guest_data['name'].replace(' ', '_')}.png")
                msg.attach(img)
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['sender_email'], email_config['sender_password'])
                server.send_message(msg)
            
            logging.info(f"Email sent successfully to {guest_data['email']}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email to {guest_data['email']}: {str(e)}")
            return False

class QRValidator:
    def __init__(self):
        self.attendance_file = "event_attendance.csv"
        self.guests_file = "guest_master_list.csv"
        self.initialize_attendance_file()
        logging.info("QRValidator initialized")
    
    def initialize_attendance_file(self):
        """Initialize attendance spreadsheet with headers"""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'guest_id', 'guest_name', 'event_name', 
                    'check_in_time', 'status', 'verified', 'notes'
                ])
            logging.info(f"Created attendance file: {self.attendance_file}")
    
    def validate_qr_code(self, qr_data):
        """Validate QR code data and extract guest information"""
        try:
            lines = qr_data.strip().split('\n')
            guest_info = {}
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    guest_info[key] = value
            
            # Validate required fields
            required_fields = ['guest_id', 'name', 'event']
            for field in required_fields:
                if field not in guest_info:
                    logging.warning(f"Missing required field in QR code: {field}")
                    return None, "Invalid QR code format"
            
            logging.info(f"QR code validated for guest: {guest_info.get('name')}")
            return guest_info, "Valid QR code"
            
        except Exception as e:
            logging.error(f"QR validation error: {e}")
            return None, f"Error: {str(e)}"
    
    def record_attendance(self, guest_info, status="Checked In", notes=""):
        """Record guest attendance in spreadsheet"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if guest already checked in today
            existing_entry = self.check_existing_entry(guest_info['guest_id'])
            
            if existing_entry and status == "Checked In":
                logging.warning(f"Guest {guest_info.get('name')} already checked in today")
                return False, "Guest already checked in today"
            
            # Add new entry
            with open(self.attendance_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp,
                    guest_info.get('guest_id', ''),
                    guest_info.get('name', ''),
                    guest_info.get('event', ''),
                    datetime.now().strftime("%H:%M:%S"),
                    status,
                    "Yes",
                    notes
                ])
            
            logging.info(f"Attendance recorded for {guest_info.get('name')}: {status}")
            return True, f"{status} successfully recorded"
            
        except Exception as e:
            logging.error(f"Failed to record attendance: {e}")
            return False, f"Error recording attendance: {str(e)}"
    
    def check_existing_entry(self, guest_id):
        """Check if guest already has an entry today"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            if os.path.exists(self.attendance_file):
                df = pd.read_csv(self.attendance_file)
                if not df.empty and 'timestamp' in df.columns:
                    today_entries = df[
                        (df['guest_id'] == guest_id) & 
                        (df['timestamp'].str.startswith(today)) &
                        (df['status'] == 'Checked In')
                    ]
                    return not today_entries.empty
            return False
        except Exception as e:
            logging.error(f"Error checking existing entry: {e}")
            return False
    
    def generate_attendance_report(self):
        """Generate attendance report"""
        try:
            if not os.path.exists(self.attendance_file):
                return "No attendance records found"
            
            df = pd.read_csv(self.attendance_file)
            
            if df.empty:
                return "No attendance records found"
            
            report = "📊 ATTENDANCE REPORT\n"
            report += "=" * 50 + "\n"
            
            # Summary statistics
            total_checkins = len(df[df['status'] == 'Checked In'])
            unique_guests = df['guest_id'].nunique()
            today = datetime.now().strftime("%Y-%m-%d")
            today_checkins = len(df[df['timestamp'].str.startswith(today)]) if 'timestamp' in df.columns else 0
            
            report += f"Total Check-ins: {total_checkins}\n"
            report += f"Unique Guests: {unique_guests}\n"
            report += f"Today's Check-ins: {today_checkins}\n"
            
            # Recent check-ins
            report += f"\n🕒 Recent Check-ins:\n"
            recent = df.tail(10)
            for _, row in recent.iterrows():
                time_str = row['timestamp'][11:16] if 'timestamp' in row else 'N/A'
                report += f"  {time_str} - {row['guest_name']} ({row['guest_id']})\n"
            
            logging.info("Attendance report generated")
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"

# ==================== GUI APPLICATION ====================

class InvitationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎉 Event Manager Pro - Invitation & QR System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f8ff')
        
        # Initialize components
        self.generator = InvitationCardGenerator()
        self.validator = QRValidator()
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_invitation_tab()
        self.create_batch_tab()
        self.create_validation_tab()
        self.create_reports_tab()
        self.create_settings_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief='sunken')
        status_bar.pack(side='bottom', fill='x')
        
        logging.info("GUI Application initialized")
        
    def create_invitation_tab(self):
        """Tab for single invitation creation"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Single Invitation")
        
        # Main frame
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left frame - Input fields
        left_frame = ttk.LabelFrame(main_frame, text="Guest Information", padding=15)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        fields = [
            ("Name:", "name"),
            ("Email:", "email"),
            ("Event Name:", "event_name"),
            ("Event Date:", "event_date"),
            ("Event Time:", "event_time"),
            ("Venue:", "venue"),
            ("RSVP Date:", "rsvp_date"),
            ("Additional Info:", "additional_info")
        ]
        
        self.entries = {}
        for i, (label, key) in enumerate(fields):
            ttk.Label(left_frame, text=label).grid(row=i, column=0, sticky='w', pady=5)
            if key == "additional_info":
                entry = scrolledtext.ScrolledText(left_frame, width=30, height=4)
                entry.grid(row=i, column=1, pady=5, padx=(10, 0))
            else:
                entry = ttk.Entry(left_frame, width=30)
                entry.grid(row=i, column=1, pady=5, padx=(10, 0))
            self.entries[key] = entry
        
        # Email configuration
        email_frame = ttk.LabelFrame(left_frame, text="Email Configuration", padding=10)
        email_frame.grid(row=len(fields), column=0, columnspan=2, sticky='we', pady=10)
        
        email_fields = [
            ("SMTP Server:", "smtp_server"),
            ("SMTP Port:", "smtp_port"),
            ("Sender Email:", "sender_email"),
            ("Password:", "sender_password")
        ]
        
        self.email_entries = {}
        for i, (label, key) in enumerate(email_fields):
            ttk.Label(email_frame, text=label).grid(row=i, column=0, sticky='w', pady=2)
            if key == "sender_password":
                entry = ttk.Entry(email_frame, width=25, show="*")
            else:
                entry = ttk.Entry(email_frame, width=25)
            entry.grid(row=i, column=1, pady=2, padx=(10, 0))
            self.email_entries[key] = entry
        
        # Set default values for email
        self.email_entries['smtp_server'].insert(0, "smtp.gmail.com")
        self.email_entries['smtp_port'].insert(0, "587")
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=len(fields)+1, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Create Invitation", 
                  command=self.create_single_invitation).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Create + Send Email", 
                  command=self.create_and_send_invitation).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear Form", 
                  command=self.clear_form).pack(side='left', padx=5)
        
        # Right frame - Preview
        right_frame = ttk.LabelFrame(main_frame, text="Preview", padding=15)
        right_frame.pack(side='right', fill='both', expand=True)
        
        self.preview_label = ttk.Label(right_frame, text="Invitation preview will appear here")
        self.preview_label.pack(expand=True)
        
        self.guest_id_label = ttk.Label(right_frame, text="", font=('Arial', 12, 'bold'))
        self.guest_id_label.pack(pady=10)
    
    def create_batch_tab(self):
        """Tab for batch invitation creation"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Batch Invitations")
        
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # CSV Import section
        csv_frame = ttk.LabelFrame(main_frame, text="CSV Import", padding=15)
        csv_frame.pack(fill='x', pady=10)
        
        ttk.Button(csv_frame, text="Import CSV File", 
                  command=self.import_csv).pack(side='left', padx=5)
        ttk.Button(csv_frame, text="Download Template", 
                  command=self.download_template).pack(side='left', padx=5)
        
        self.csv_status = ttk.Label(csv_frame, text="No CSV file selected")
        self.csv_status.pack(side='left', padx=20)
        
        # Email configuration for batch
        email_frame = ttk.LabelFrame(main_frame, text="Batch Email Configuration", padding=10)
        email_frame.pack(fill='x', pady=10)
        
        ttk.Label(email_frame, text="SMTP Server:").grid(row=0, column=0, sticky='w')
        self.batch_smtp = ttk.Entry(email_frame, width=20)
        self.batch_smtp.grid(row=0, column=1, padx=5)
        self.batch_smtp.insert(0, "smtp.gmail.com")
        
        ttk.Label(email_frame, text="Port:").grid(row=0, column=2, sticky='w', padx=(10,0))
        self.batch_port = ttk.Entry(email_frame, width=10)
        self.batch_port.grid(row=0, column=3, padx=5)
        self.batch_port.insert(0, "587")
        
        ttk.Label(email_frame, text="Email:").grid(row=1, column=0, sticky='w', pady=5)
        self.batch_email = ttk.Entry(email_frame, width=20)
        self.batch_email.grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(email_frame, text="Password:").grid(row=1, column=2, sticky='w', pady=5, padx=(10,0))
        self.batch_password = ttk.Entry(email_frame, width=15, show="*")
        self.batch_password.grid(row=1, column=3, pady=5, padx=5)
        
        # Batch buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Create Invitations Only", 
                  command=self.batch_create_only).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Create + Send Emails", 
                  command=self.batch_create_and_send).pack(side='left', padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.pack(fill='x', pady=10)
        
        self.progress_label = ttk.Label(main_frame, text="")
        self.progress_label.pack()
    
    def create_validation_tab(self):
        """Tab for QR code validation"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="QR Validation")
        
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Scanner controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=10)
        
        ttk.Button(control_frame, text="Start Scanner", 
                  command=self.start_scanner).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Stop Scanner", 
                  command=self.stop_scanner).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Manual Entry", 
                  command=self.manual_validation).pack(side='left', padx=5)
        
        # Camera feed
        self.camera_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        self.camera_frame.pack(fill='both', expand=True, pady=10)
        
        self.camera_label = ttk.Label(self.camera_frame, text="Camera feed will appear here\n\nClick 'Start Scanner' to begin")
        self.camera_label.pack(expand=True)
        
        # Add warning if no QR decoder available
        if not ZBAR_AVAILABLE and (not hasattr(simple_decoder, 'available') or not simple_decoder.available):
            warning_label = ttk.Label(self.camera_frame, text="⚠️ QR scanning disabled - No decoder available", 
                                    foreground='red')
            warning_label.pack(side='bottom', pady=5)
        
        # Validation results
        result_frame = ttk.LabelFrame(main_frame, text="Validation Results", padding=10)
        result_frame.pack(fill='x', pady=10)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=8)
        self.result_text.pack(fill='both', expand=True)
        
        self.scanning = False
        self.cap = None
    
    def create_reports_tab(self):
        """Tab for reports and analytics"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Reports & Analytics")
        
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Report controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=10)
        
        ttk.Button(control_frame, text="Generate Attendance Report", 
                  command=self.generate_report).pack(side='left', padx=5)
        ttk.Button(control_frame, text="View Current Attendance", 
                  command=self.view_attendance).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export to Excel", 
                  command=self.export_excel).pack(side='left', padx=5)
        
        # Report display
        report_frame = ttk.LabelFrame(main_frame, text="Report Output", padding=10)
        report_frame.pack(fill='both', expand=True, pady=10)
        
        self.report_text = scrolledtext.ScrolledText(report_frame, height=15)
        self.report_text.pack(fill='both', expand=True)
    
    def create_settings_tab(self):
        """Tab for application settings"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Settings")
        
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Output directory
        dir_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding=10)
        dir_frame.pack(fill='x', pady=10)
        
        ttk.Label(dir_frame, text="Output Directory:").grid(row=0, column=0, sticky='w')
        self.dir_var = tk.StringVar(value="invitation_cards")
        ttk.Entry(dir_frame, textvariable=self.dir_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2)
        
        # Default email settings
        email_frame = ttk.LabelFrame(main_frame, text="Default Email Settings", padding=10)
        email_frame.pack(fill='x', pady=10)
        
        ttk.Button(email_frame, text="Save Default Settings", 
                  command=self.save_settings).pack(pady=5)
        
        # About section
        about_frame = ttk.LabelFrame(main_frame, text="About", padding=10)
        about_frame.pack(fill='x', pady=10)
        
        about_text = """Event Manager Pro v1.0
A complete invitation and QR validation system
        
Features:
• Create beautiful invitation cards with QR codes
• Send invitations via email
• QR code validation and attendance tracking
• Batch processing from CSV files
• Comprehensive reporting
        
Created with Python and Tkinter"""
        
        ttk.Label(about_frame, text=about_text, justify='left').pack(anchor='w')
    
    # Core functionality methods
    def create_single_invitation(self):
        try:
            guest_data = self.get_guest_data()
            if not guest_data['name']:
                messagebox.showerror("Error", "Please enter guest name")
                return
            
            card_filename, guest_id = self.generator.create_invitation_card(guest_data)
            self.show_preview(card_filename)
            self.guest_id_label.config(text=f"Guest ID: {guest_id}")
            self.status_var.set(f"Invitation created for {guest_data['name']}")
            messagebox.showinfo("Success", f"Invitation created successfully!\nGuest ID: {guest_id}")
            
        except Exception as e:
            logging.error(f"Failed to create invitation: {e}")
            messagebox.showerror("Error", f"Failed to create invitation: {str(e)}")
    
    def create_and_send_invitation(self):
        try:
            guest_data = self.get_guest_data()
            email_config = self.get_email_config()
            
            if not guest_data['name'] or not guest_data.get('email'):
                messagebox.showerror("Error", "Please enter guest name and email")
                return
            
            if not email_config['sender_email'] or not email_config['sender_password']:
                messagebox.showerror("Error", "Please enter email configuration")
                return
            
            card_filename, guest_id = self.generator.create_invitation_card(guest_data)
            success = self.generator.send_invitation_email(guest_data, card_filename, guest_id, email_config)
            
            if success:
                self.show_preview(card_filename)
                self.guest_id_label.config(text=f"Guest ID: {guest_id}")
                self.status_var.set(f"Invitation sent to {guest_data['name']}")
                messagebox.showinfo("Success", f"Invitation sent successfully!\nGuest ID: {guest_id}")
            else:
                messagebox.showerror("Error", "Failed to send email. Check your email configuration.")
            
        except Exception as e:
            logging.error(f"Failed to send invitation: {e}")
            messagebox.showerror("Error", f"Failed to send invitation: {str(e)}")
    
    def batch_create_only(self):
        if not hasattr(self, 'batch_df'):
            messagebox.showerror("Error", "Please import a CSV file first")
            return
        
        threading.Thread(target=self._batch_create_only, daemon=True).start()
    
    def _batch_create_only(self):
        try:
            total = len(self.batch_df)
            self.progress['value'] = 0
            self.progress['maximum'] = total
            
            for i, (_, row) in enumerate(self.batch_df.iterrows()):
                guest_data = {
                    'name': row['name'],
                    'event_name': row.get('event_name', 'Event'),
                    'event_date': row.get('event_date', 'TBD'),
                    'event_time': row.get('event_time', 'TBD'),
                    'venue': row.get('venue', 'TBD'),
                    'rsvp_date': row.get('rsvp_date', 'TBD'),
                    'additional_info': row.get('additional_info', ''),
                    'email': row.get('email', '')
                }
                self.generator.create_invitation_card(guest_data)
                
                # Update progress
                self.progress['value'] = i + 1
                self.progress_label.config(text=f"Processed {i+1}/{total}")
                self.root.update_idletasks()
            
            self.status_var.set(f"Batch creation completed: {total} invitations")
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Created {total} invitations"))
            
        except Exception as e:
            logging.error(f"Batch creation failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Batch creation failed: {str(e)}"))
    
    def batch_create_and_send(self):
        if not hasattr(self, 'batch_df'):
            messagebox.showerror("Error", "Please import a CSV file first")
            return
        
        email_config = {
            'smtp_server': self.batch_smtp.get(),
            'smtp_port': int(self.batch_port.get() or 587),
            'sender_email': self.batch_email.get(),
            'sender_password': self.batch_password.get()
        }
        
        if not email_config['sender_email'] or not email_config['sender_password']:
            messagebox.showerror("Error", "Please enter email configuration")
            return
        
        threading.Thread(target=self._batch_create_and_send, args=(email_config,), daemon=True).start()
    
    def _batch_create_and_send(self, email_config):
        try:
            total = len(self.batch_df)
            self.progress['value'] = 0
            self.progress['maximum'] = total
            sent_count = 0
            
            for i, (_, row) in enumerate(self.batch_df.iterrows()):
                guest_data = {
                    'name': row['name'],
                    'event_name': row.get('event_name', 'Event'),
                    'event_date': row.get('event_date', 'TBD'),
                    'event_time': row.get('event_time', 'TBD'),
                    'venue': row.get('venue', 'TBD'),
                    'rsvp_date': row.get('rsvp_date', 'TBD'),
                    'additional_info': row.get('additional_info', ''),
                    'email': row.get('email', '')
                }
                
                if guest_data['email']:
                    card_filename, guest_id = self.generator.create_invitation_card(guest_data)
                    if self.generator.send_invitation_email(guest_data, card_filename, guest_id, email_config):
                        sent_count += 1
                else:
                    self.generator.create_invitation_card(guest_data)
                
                # Update progress
                self.progress['value'] = i + 1
                self.progress_label.config(text=f"Processed {i+1}/{total} (Sent: {sent_count})")
                self.root.update_idletasks()
            
            self.status_var.set(f"Batch completed: {total} created, {sent_count} sent")
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                f"Created {total} invitations\nSent {sent_count} emails"))
            
        except Exception as e:
            logging.error(f"Batch operation failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Batch operation failed: {str(e)}"))
    
    def start_scanner(self):
        # Check if QR decoder is available
        if not ZBAR_AVAILABLE and (not hasattr(simple_decoder, 'available') or not simple_decoder.available):
            messagebox.showerror("Error", "QR scanning is not available. Please install OpenCV or pyzbar.")
            return
        
        self.scanning = True
        self.camera_label.config(text="Initializing camera...")
        threading.Thread(target=self._scan_qr_codes, daemon=True).start()
    
    def _scan_qr_codes(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # Try different camera indices
                for i in range(3):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                
                if not self.cap.isOpened():
                    self.camera_label.after(0, lambda: self.camera_label.config(
                        text="Error: Cannot access any camera\n\nClick 'Start Scanner' to retry"))
                    logging.error("Cannot access any camera")
                    return
        except Exception as e:
            self.camera_label.after(0, lambda: self.camera_label.config(
                text=f"Camera error: {str(e)}\n\nClick 'Start Scanner' to retry"))
            logging.error(f"Camera initialization error: {e}")
            return
        
        self.camera_label.after(0, lambda: self.camera_label.config(
            text="Camera active - Scanning for QR codes..."))
        
        while self.scanning:
            try:
                ret, frame = self.cap.read()
                if ret:
                    decoded_objects = QRDecoder(frame)
                    for obj in decoded_objects:
                        self.process_qr_code(obj.data.decode('utf-8'), frame)
                    
                    # Display frame
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (640, 480))
                        img = Image.fromarray(frame)
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.camera_label.after(0, lambda img=imgtk: self.update_camera_feed(img))
                    except Exception as e:
                        pass
                
                # Small delay to prevent high CPU usage
                cv2.waitKey(1)
                
            except Exception as e:
                logging.error(f"Error during QR scanning: {e}")
                break
        
        if self.cap:
            self.cap.release()
        self.camera_label.after(0, lambda: self.camera_label.config(
            text="Camera stopped\n\nClick 'Start Scanner' to begin"))
    
    def update_camera_feed(self, imgtk):
        """Thread-safe camera feed update"""
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)
    
    def process_qr_code(self, qr_data, frame):
        guest_info, message = self.validator.validate_qr_code(qr_data)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if guest_info:
            success, result_msg = self.validator.record_attendance(guest_info)
            if success:
                result_text = f"[{timestamp}] ✅ {guest_info.get('name')} - {result_msg}\n"
                self.status_var.set(f"Checked in: {guest_info.get('name')}")
            else:
                result_text = f"[{timestamp}] ⚠️ {guest_info.get('name')} - {result_msg}\n"
        else:
            result_text = f"[{timestamp}] ❌ {message}\n"
        
        self.result_text.after(0, lambda txt=result_text: self.update_result_text(txt))
    
    def update_result_text(self, text):
        """Thread-safe result text update"""
        self.result_text.insert('end', text)
        self.result_text.see('end')
    
    def stop_scanner(self):
        self.scanning = False
        if self.cap:
            self.cap.release()
    
    def manual_validation(self):
        manual_window = tk.Toplevel(self.root)
        manual_window.title("Manual Validation")
        manual_window.geometry("400x300")
        
        ttk.Label(manual_window, text="Enter QR Code Data:").pack(pady=10)
        qr_text = scrolledtext.ScrolledText(manual_window, height=10)
        qr_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        def validate_manual():
            data = qr_text.get('1.0', 'end').strip()
            if data:
                guest_info, message = self.validator.validate_qr_code(data)
                if guest_info:
                    success, result_msg = self.validator.record_attendance(guest_info)
                    if success:
                        messagebox.showinfo("Success", f"Checked in: {guest_info.get('name')}\n{result_msg}")
                    else:
                        messagebox.showwarning("Warning", f"{guest_info.get('name')} - {result_msg}")
                else:
                    messagebox.showerror("Error", message)
                manual_window.destroy()
        
        ttk.Button(manual_window, text="Validate", command=validate_manual).pack(pady=10)
    
    def generate_report(self):
        try:
            report = self.validator.generate_attendance_report()
            self.report_text.delete('1.0', 'end')
            self.report_text.insert('1.0', report)
            self.status_var.set("Report generated successfully")
        except Exception as e:
            logging.error(f"Failed to generate report: {e}")
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def view_attendance(self):
        try:
            if not os.path.exists(self.validator.attendance_file):
                self.report_text.delete('1.0', 'end')
                self.report_text.insert('1.0', "No attendance records found")
                return
            
            df = pd.read_csv(self.validator.attendance_file)
            if df.empty:
                self.report_text.delete('1.0', 'end')
                self.report_text.insert('1.0', "No attendance records found")
                return
            
            report = "📊 CURRENT ATTENDANCE\n" + "="*50 + "\n"
            for _, row in df.tail(20).iterrows():
                time_str = row['timestamp'][11:16] if 'timestamp' in row else 'N/A'
                report += f"{time_str} - {row['guest_name']} ({row['guest_id']})\n"
            
            self.report_text.delete('1.0', 'end')
            self.report_text.insert('1.0', report)
            self.status_var.set("Attendance data loaded")
            
        except Exception as e:
            logging.error(f"Failed to load attendance: {e}")
            messagebox.showerror("Error", f"Failed to load attendance: {str(e)}")
    
    def export_excel(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if filename:
                if os.path.exists(self.validator.attendance_file):
                    df = pd.read_csv(self.validator.attendance_file)
                    df.to_excel(filename, index=False)
                    messagebox.showinfo("Success", f"Data exported to {filename}")
                    self.status_var.set(f"Data exported to {filename}")
                else:
                    messagebox.showerror("Error", "No attendance data to export")
        except Exception as e:
            logging.error(f"Export failed: {e}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def import_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.batch_df = pd.read_csv(filename)
                self.csv_status.config(text=f"Loaded {len(self.batch_df)} guests")
                self.status_var.set(f"CSV imported: {len(self.batch_df)} guests")
                messagebox.showinfo("Success", f"Successfully imported {len(self.batch_df)} guests")
                logging.info(f"CSV imported: {filename} with {len(self.batch_df)} guests")
            except Exception as e:
                logging.error(f"Failed to import CSV: {e}")
                messagebox.showerror("Error", f"Failed to import CSV: {str(e)}")
    
    def download_template(self):
        template_data = [
            ['name', 'email', 'event_name', 'event_date', 'event_time', 'venue', 'rsvp_date', 'additional_info'],
            ['John Doe', 'john@example.com', 'Wedding Celebration', 'June 15, 2024', '2:00 PM', 'Grand Hotel', 'June 1, 2024', 'Black tie event'],
            ['Jane Smith', 'jane@example.com', 'Wedding Celebration', 'June 15, 2024', '2:00 PM', 'Grand Hotel', 'June 1, 2024', 'Plus one allowed']
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="guest_template.csv"
        )
        if filename:
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(template_data)
            messagebox.showinfo("Success", f"Template saved as {filename}")
            self.status_var.set("CSV template downloaded")
            logging.info(f"Template downloaded: {filename}")
    
    def show_preview(self, image_path):
        try:
            image = Image.open(image_path)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
        except Exception as e:
            logging.error(f"Failed to show preview: {e}")
            self.preview_label.configure(text="Preview not available")
    
    def get_guest_data(self):
        data = {}
        for key, entry in self.entries.items():
            if key == "additional_info":
                data[key] = entry.get('1.0', 'end').strip()
            else:
                data[key] = entry.get().strip()
        return data
    
    def get_email_config(self):
        return {
            'smtp_server': self.email_entries['smtp_server'].get(),
            'smtp_port': int(self.email_entries['smtp_port'].get() or 587),
            'sender_email': self.email_entries['sender_email'].get(),
            'sender_password': self.email_entries['sender_password'].get()
        }
    
    def clear_form(self):
        for key, entry in self.entries.items():
            if key == "additional_info":
                entry.delete('1.0', 'end')
            else:
                entry.delete(0, 'end')
        self.guest_id_label.config(text="")
        self.preview_label.config(text="Invitation preview will appear here", image='')
    
    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_var.set(directory)
            self.generator.output_dir = directory
            self.status_var.set(f"Output directory set to: {directory}")
            logging.info(f"Output directory changed to: {directory}")
    
    def save_settings(self):
        # In a real application, you would save these to a config file
        messagebox.showinfo("Settings", "Settings saved successfully")
        self.status_var.set("Settings saved")
        logging.info("Settings saved")
    
    def on_closing(self):
        self.stop_scanner()
        logging.info("Application closing...")
        self.root.destroy()

def main():
    # Check packages first (non-blocking)
    all_packages_ok = check_packages()
    
    root = tk.Tk()
    app = InvitationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center the window
    root.update_idletasks()
    width = 1200
    height = 800
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Show startup info
    print("\n" + "="*50)
    print("Event Manager Pro GUI Initialized")
    print("="*50 + "\n")
    
    if not all_packages_ok:
        print("⚠️  Some required packages are missing.")
        print("   The application may have limited functionality.")
        print("   Check the console output above for installation instructions.\n")
    
    root.mainloop()

if __name__ == "__main__":
    main()