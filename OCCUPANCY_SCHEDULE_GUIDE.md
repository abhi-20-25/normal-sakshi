# 📋 OccupancyMonitor Schedule Upload Guide

## 🎯 Overview

The OccupancyMonitor now supports **CSV/Excel file upload** for easy schedule management. You can upload a schedule file instead of manually entering data in the grid interface.

## 🚀 Features

- ✅ **CSV Upload**: Upload `.csv` files with schedule data
- ✅ **Excel Upload**: Upload `.xlsx` and `.xls` files
- ✅ **Template Download**: Get pre-formatted template files
- ✅ **Auto-Apply**: Schedule is automatically applied after upload
- ✅ **Validation**: File format and data validation
- ✅ **Database Integration**: Schedules saved to PostgreSQL
- ✅ **Real-time Updates**: Changes applied immediately

## 📁 File Format

### CSV Template Structure
```csv
Day,00:00,01:00,02:00,03:00,04:00,05:00,06:00,07:00,08:00,09:00,10:00,11:00,12:00,13:00,14:00,15:00,16:00,17:00,18:00,19:00,20:00,21:00,22:00,23:00
Monday,0,0,0,0,0,0,0,0,2,3,4,4,3,2,2,3,4,4,3,2,1,0,0,0
Tuesday,0,0,0,0,0,0,0,0,2,3,4,4,3,2,2,3,4,4,3,2,1,0,0,0
Wednesday,0,0,0,0,0,0,0,0,2,3,4,4,3,2,2,3,4,4,3,2,1,0,0,0
Thursday,0,0,0,0,0,0,0,0,2,3,4,4,3,2,2,3,4,4,3,2,1,0,0,0
Friday,0,0,0,0,0,0,0,0,2,3,4,4,3,2,2,3,4,4,3,2,1,0,0,0
Saturday,0,0,0,0,0,0,0,0,1,2,3,3,2,1,1,2,3,3,2,1,0,0,0,0
Sunday,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0
```

### Excel Template Structure
Same as CSV but in Excel format with:
- **Row 1**: Days (Monday, Tuesday, etc.)
- **Column 1**: Time slots (00:00, 01:00, etc.)
- **Values**: Required occupancy count (0-100)

## 📋 How to Use

### Step 1: Access Schedule Management
1. Go to the **OccupancyMonitor** section in the dashboard
2. Click **"Manage Schedule"** button
3. The schedule management modal will open

### Step 2: Download Template
1. Click **"Download Template"** button
2. A CSV file will be downloaded with the correct format
3. Open the file in Excel, Google Sheets, or any CSV editor

### Step 3: Edit Schedule
1. **Edit the numbers** in the template:
   - `0` = No occupancy required
   - `1-100` = Required number of people
2. **Save the file** as CSV or Excel format
3. **Keep the format** exactly as downloaded (don't change column headers)

### Step 4: Upload Schedule
1. Click **"Choose File"** and select your edited file
2. Click **"Upload & Apply"** button
3. The schedule will be processed and applied automatically
4. You'll see a success message with the number of time slots updated

### Step 5: Verify Schedule
1. The **"Current Schedule"** section will show your uploaded schedule
2. You can still make **manual changes** if needed
3. Click **"Save Manual Changes"** to apply any manual edits

## 🎯 Schedule Examples

### Business Hours Schedule
- **Weekdays (Mon-Fri)**: 2-4 people during 8 AM - 8 PM
- **Weekends (Sat-Sun)**: 1-2 people during 9 AM - 6 PM
- **Off Hours**: 0 people

### 24/7 Schedule
- **Always**: 1-2 people minimum
- **Peak Hours**: 3-5 people during 10 AM - 6 PM
- **Night Shift**: 1 person during 10 PM - 6 AM

### Custom Schedule
- **Monday**: 3 people during 9 AM - 5 PM
- **Tuesday**: 2 people during 10 AM - 4 PM
- **Wednesday**: 4 people during 8 AM - 6 PM
- **Thursday**: 2 people during 9 AM - 5 PM
- **Friday**: 3 people during 8 AM - 5 PM
- **Saturday**: 1 person during 10 AM - 2 PM
- **Sunday**: 0 people (closed)

## ⚠️ Important Notes

### File Requirements
- **Format**: CSV (.csv) or Excel (.xlsx, .xls)
- **Headers**: Must have "Day" column and time columns (HH:MM format)
- **Days**: Must be exact: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
- **Times**: Must be in 24-hour format (00:00, 01:00, etc.)
- **Values**: Must be integers between 0-100

### Validation Rules
- ✅ File must have "Day" column
- ✅ Time columns must be in HH:MM format
- ✅ Day names must match exactly
- ✅ Values must be integers (0-100)
- ❌ Empty cells are treated as 0
- ❌ Invalid formats are rejected

### Error Handling
- **File not selected**: "Please select a file to upload"
- **Invalid format**: "Please select a CSV or Excel file"
- **Missing Day column**: "CSV must have 'Day' column"
- **Upload error**: "Error uploading schedule. Please try again."

## 🔧 Technical Details

### API Endpoints
- `GET /api/occupancy/schedule/template` - Download CSV template
- `POST /api/occupancy/schedule/upload/{channel_id}` - Upload schedule file
- `GET /api/occupancy/schedule/{channel_id}` - Get current schedule
- `POST /api/occupancy/schedule/{channel_id}` - Update schedule manually

### Database Tables
- `occupancy_schedules` - Stores schedule data
- `occupancy_logs` - Stores monitoring logs

### File Processing
1. **Upload**: File is received via multipart form data
2. **Parse**: pandas library reads CSV/Excel data
3. **Validate**: Format and data validation
4. **Process**: Convert to schedule format
5. **Save**: Store in PostgreSQL database
6. **Apply**: Update OccupancyMonitor processor

## 🎉 Benefits

- **Easy Management**: No more manual grid editing
- **Bulk Updates**: Set entire week schedule at once
- **Template System**: Pre-formatted files for consistency
- **Validation**: Automatic error checking
- **Flexibility**: Support for both CSV and Excel
- **Integration**: Seamless database integration
- **Real-time**: Changes applied immediately

## 🚀 Quick Start

1. **Download** the template CSV file
2. **Edit** the occupancy numbers for each day/hour
3. **Upload** the file through the interface
4. **Schedule** is automatically applied and active!

The OccupancyMonitor will now use your uploaded schedule for monitoring and alerting! 🎯
