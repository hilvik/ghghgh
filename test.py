from supabase import create_client

supabase_url = "https://emefyicilkiaaqkbjsjy.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWZ5aWNpbGtpYWFxa2Jqc2p5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUzMzMzMTgsImV4cCI6MjA2MDkwOTMxOH0.UDQCvVb4zLa1wS_CyxpNH0WN58SCwOtLrK5jlMNXc5I"

try:
    supabase = create_client(supabase_url, supabase_key)
    result = supabase.table("topics").select("*").execute()
    print("Connection successful!")
    print(f"Retrieved {len(result.data)} records from topics table")
except Exception as e:
    print(f"Connection failed: {str(e)}")