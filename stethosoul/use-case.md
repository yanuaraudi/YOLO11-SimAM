# Use Case Scenario Web Stethosoul

## Registration
### Use Case ID : UC-001
**Use Case Name**: Registrasi Admin Sekolah  
**Actors** : Admin Sekolah  
**Description** : Admin sekolah mendaftarkan institusinya (Sekolah/Perguruan Tinggi) ke dalam sistem untuk mendapatkan dashboard dan subdomain khusus.  
**Preconditions** : Admin berada di halaman utama pendaftaran untuk sekolah.  
**Postconditions** : Akun admin sekolah berhasil dibuat, dan subdomain sekolah (misal: ```stethosoul.id/namasekolah``` telah aktif dan bisa diakses.  
**Trigger** : Admin memilih pilihan "Akun Sekolah/Kampus" di pilihan pembuatan akun.

**Main Flow (Basic Path)**  
1. Admin masuk ke laman pemilihan pembuatan akun
2. Admin memilih untuk membuat akun Sekolah/Kampus
3. Admin mengisi formulir pendaftaran yang berisi:
    - Nama Institusi
    - Jenis Institusi
    - Alamat Lengkap Sekolah
    - Nama Penanggung Jawab
    - Email Penanggung Jawab
    - Password
4. Admin menekan tombol "Daftar"
5. Sistem memvalidasi data yang dimasukkan
6. Sistem mengirimkan OTP ke email yang dimasukkan oleh Admin
7. Admin mengisi kode OTP
8. Sistem memverifikasi Email
9. Sistem membuat akun Sekolah
10. Sistem secara otomatis membuat subdomain unik berdasarkan nama sekolah.
11. Sistem mengarahkan admin ke halaman login dan menampilkan notifikasi bahwa pendaftaran berhasil.

**Alternative Flow (Optional Path)**  
- Tidak Ada.

**Exception Flow (Error Handling)**  
- **E1**: Jika nama subdomain sudah ada -> Sistem menampilakn pesan "Nama sekolah sudah terdaftar, silahkan gunakan nama lain".
- **E2**: Jika format email tidak valid -> Sistem menampilkan pesan "Format email tidak valid."
- **E3**: Jika ada field wajib yang kosong -> Sistem menampilkan pesan error di bawah field yang bersangkutan.

### Use Case ID : UC-002
**Use Case Name** : Registrasi Siswa/Mahasiswa  
**Actors** : Siswa/Mahasiswa  
**Description** : Siswa atau Mahasiswa mendaftarkan diri melalui subdomain sekolah mereka yang sudah terdaftar untuk dapat melakukan screening.  
**Preconditions** : Admin Sekolah sudah mendaftarkan sekolah dan memberikan link subdomain pendaftaran kepada Siswa/Mahasiswa (misal: ```stethosoul.di/namasekolah/daftar```).  
**Postconditions**: Akun Siswa/Mahasiswa berhasil dibuat dan terasosiasi dengan sekolahnya.  
**Trigger** : Siswa/Mahasiswa mengakses halaman pendaftaran akun Siswa/Mahasiswa melalui subdomain sekolah.  

**Main Flow (Basic Path)**
1. Siswa mengakses URL pendaftaran subdomain sekolahnya.
2. Siswa mengisi formulir pendaftaran:
    - Nama Lengkap
    - Tanggal Lahir
    - NIM/NISN
    - Email Aktif
    - Nama Orang Tua/Wali
    - Surat Persetujuan Orang Tua
    - Password
3. Siswa (khusus jenjang SMA/SMK) mengunggah file surat persetujuan orang tua.
4. Siswa menekan tombol "Daftar".
5. Sistem memvalidasi data dan file yang diunggah.
6. Sistem mengirimkan OTP ke email yang dimasukkan oleh Siswa.
7. Siswa mengisi kode OTP.
8. Sistem memverifikasi Email.
9. Sistem membuat akun Siswa yang terhubung dengan sekolah tersebut.
10. Sistem mengarahkan Siswa ke halaman login subdomain dan menampilkan notifikasi bahwa pendaftaran berhasil.

**Alternative Flow (Optional Path)**
- **A1**: Jika siswa adalah mahasiswa (Bukan SMA/SMK) -> Langkah untuk menunggah surat persetujuan tidak ditampilkan.
- **A2**: Admin bisa membuatkan akun untuk Siswa

**Exception Flow (Error Handling)**
- **E1**: Jika NIM/NISN sudah terdaftar -> Sistem menampilkan pesan "NIM/NISN sudah terdaftar."
- **E2**: Jika email sudah terdaftar -> Sistem menampilkan pesan "Email sudah terdaftar."
- **E3**: Jika siswa SMA/SMK tidak mengunggah surat persetujuan -> Sistem menampilkan pesan "Anda wajib mengunggah surat persetujuan orang tua."
- **E4**: Jika format file yang diunggah salah (Bukan PDF/DOCX/JPG) -> Sistem menampilakn pesan "Format file tidak valid. Harap unggah dalam format PDF/DOCX/JPG." 

### Use Case ID : UC-003

**Use Case Name** : Registrasi User Umum/Publik  
**Actors** : User Umum/Publik  
**Description** : Pengguna umum (di luat sekolah terdaftar) membuat akun untuk dapat melakukan screening kesehatan mental dan melihat riwayatnya.  
**Preconditions** : Pengguna berada di halaman utama pendaftaran publik.  
**Postconditions** : Akun pengguna umum berhasil dibuat dan pengguna masuk ke dashboard.  
**Trigger** : Pengguna meng-klik tombol daftar.

**Main Flow (Basic Path)**
1. Pengguna mengisi formulir pendaftaran:
    - Nama Lengkap
    - Tanggal Lahir
    - Email Aktif
    - Password
    - Surat Persetujuan Orang Tua/Wali
2. Pengguna menekan tombol "Daftar"
3. Sistem memvalidasi data
4. Sistem mengirimkan OTP ke email yang dimasukkan oleh pengguna
5. Pengguna mengisi kode OTP
6. Sistem memverifikasi Email
7. Sistem membuat akun baru.
8. Sistem mengarahkan pengguna ke halaman login dan menampilkan notifikasi bahwa pendaftaran berhasil.

**Alternative Flow (Optional Path)**
- **A1** : Pendaftaran Google ->
    1. Pengguna memilih opsi "Lanjutkan dengan Google".
    2. Sistem mengarahkan ke halaman autentikasi Google.
    3. Pengguna memilih akun Google yang akan digunakan.
    4. Sistem menerima data dari Google dan secara otomatis membuat akun baru di sistem.
    5. Sistem mengarahkan pengguna ke halaman dashboard.
- **A2** : Jika pengguna tidak berusia dibawah 18 tahun -> Langkah untuk mengunggah surat persetujuan tidak ditampilkan.

**Exception Flow (Error Handling)**  
- **E1** : Email sudah terdaftar -> Sistem menampilkan pesan "Email ini sudah terdaftar. Silahkan login."
- **E2** : Gagal terhubung dengan layanan Google -> Sistem menampilakn pesan "Gagal terhubung dengan Google. Silakan coba lagi atau daftar dengan mengisi formulir."

## Login Authentication

### Use Case ID : UC-004

**Use Case Name** : Login Pengguna  
**Actors** : Admin Sekolah, Siswa/Mahasiswa, User Umum  
**Description** : Pengguna (semua jenis) masuk ke dalam website menggunakan email dan password terdaftar.  
**Preconditions** : Pengguna sudah memiliki akun yang terdaftar.  
**Postconditions** : Pengguna berhasil masuk dan diarahkan ke dashboard yang sesuai dengan rolenya.  
**Trigger** : Pengguna meng-klik tombol "Masuk"  

**Main Flow (Basic Path)**  
1. pengguna mengakses halaman login yang sesuai (login utama untuk User Umum & Admin, login subdomain untuk Siswa/Mahasiswa).
2. Pengguna mengisi email.
3. Pengguna mengisi password.
4. Pengguna menekan tombol "Masuk".
5. Sistem memvalidasi email dan password.
6. Sistem memberikan akses dan mengarahkan pengguna ke halaman yang sesuai:
    - Admin Sekolah -> Dashboard Sekolah.
    - Siswa/Mahasiswa & User Umum -> Dashboard Pengguna.

**Alternative Flow (Optional Path)**
- **A1** : Login dengan Google (hanya untuk User Umum) -> Pengguna meng-klik "Masuk dengan Google" dan mengikuti alur autentikasi Google.
- **A2** : Sesi masih aktif -> Jika pengguna sudah login sebelumnya dan token sesi masih valid, sistem akan langsung mengarahkan ke halaman dashboard tanpa perlu login ulang.

**Exception Flow (Error Handling)**
- **E1** : Email/Password salah -> Sistem menampilkan pesan "Email atau password tidak valid"  
- **E2** : Akun belum terdaftar -> Sistem menampilkan pesan "Akun Anda tidak ditemukan"  
- **E3** : Sesi Habis -> Jika token valid/kadaluarsa saat mengakses halaman lain, sistem akan mengarahkan kembali ke halaman login dengan pesan "Sesi Anda telah berakhir. Mohon login ulang."

## Screening

### Use Case ID : UC-005

**Use Case Name** : Melakukan Screening  
**Actors** : Siswa/Mahasiswa, User Umum  
**Description** : Pengguna yang sudah login melakukan tes screening psikosis dini (StethoSoul) atau depresi & kecemasan (StethoSoul+).  
**Preconditions** : Pengguna telah berhasil login ke dalam sistem  
**Postconditions** : Hasil screening tersimpan di database, dan dikirimkan ke email pengguna.
**Trigger** : Pengguna memilih jenis screening dan memulai tes.

**Main Flow (Basic Path)**
1. Dari dashboard, pengguna meng-klik mulai tes. Lalu pengguna memilih jenis screening yang akan dilakukan (StethoSoul, StethoSoul+, atau keduanya sekaligus).
2. Sistem menampilkan modul screening dengan serangkaian pertanyaan.
3. Pengguna menjawab semua pertanyaan yang diberikan.
4. Setelah selesai, pengguna menekan tombol "Lihat Hasil".
5. Sistem memproses jawaban dan mengkalkulasi hasil screening (risiko rendah, sedang, tinggi).
6. Sistem menyimpan hasil screening ke riwayat tes pengguna.
7. Sistem secara otomatis mengirimkan detail hasil ke email pengguna yang terdaftar.

**Alternative flow (Optional Path)**
- Tidak ada

**Exception Flow (Error Handling)**
- **E1** : Koneksi terputus saat screening -> Sistem menyimpan progres jawaban sementara. Saat koneksi pulih, pengguna dapat melanjutkan dari pertanyaan terakhir.
- **E2** : Pengguna mencoba lanjut ke pertanyaan selanjutnya dengan pertanyaan yang sebelumnya belum dijawab -> Sistem menampilan pesan "Harap jawab pertanyaan terlebih dahulu".
- **E3** : Gagal mengirim email -> Sistem menampilkan notifikasi "Gagal mengirim hasil ke email" di riwayat screening.

## Admin Activities

### Use Case ID : UC-006

**Use Case Name** : Melihat Dashboard Laporan Sekolah  
**Actors** : Admin Sekolah  
**Description** : Admin Sekolah melihat rekapitulasi (agregat) hasil screening seluruh siswa serta dapat mencari dan melihat hasil individu per siswa  
**Precondictions** : Admin sekolah telah login  
**Postconditions** : Admin mendapatkan informasi mengenai kondisi kesehatan mental siswa di institusinya.  
**Trigger** : Admin meng-klik menu "Dashboard" atau "Manajemen Siswa"

**Main Flow (Basic Path)**
1. Admin masuk ke halaman dashboard sekolah
2. Sistem menampilkan data agregat hasil screening seluruh Siswa/Mahasiswa
3. Sistem menampilkan daftar seluruh Siswa/Mahasiswa yang terdaftar.
4. Admin menggunakan fitur pencarian dengan mengetikkan nama atau NIM/NISN siswa
5. Sistem menampilkan hasil pencarian yang cocok.
6. Admin meng-klik salah satu nama siswa dari daftar hasil pencarian.
7. Sistem menampilkan detail profil siswa beserta riwayat dan hasil screening yang pernah dilakukan.

**Alternative Flow (Optional Path)**
- **A1** : Admin menggunakan filter untuk menyortir siswa berdasarkan level risiko (rendah, sedang, tinggi)

**Exception Flow (Error Handling)**
- **E1** : Siswa tidak ditemukan -> Sistem menampilkan pesan "Siswa dengan kriteria tersebut tidak ditemukan"


### Use Case ID : UC-007

**Use Case Name** : Menambahkan Akun Siswa/Mahasiswa  
**Actors** : Admin Sekolah  
**Description** : Admin Sekolah menambahkan akun Siswa/Mahasiswa secara manual ataupun menyediakan link pendaftaran (misal: ```stethosoul.id/namasekolah/daftar```)
**Preconditions** : Admin Sekolah telah login  
**Postconditions** : Akun Siswa/Mahasiswa  
**Trigger** : Admin meng-klik menu "Manajemen Akun"

**Main Flow (Basic Path)**
1. Admin masuk ke halaman Manajemen Akun
2. Sistem menampilkan seluruh akun yang terdaftar dan terafiliasi dengan sekolah
3. Sistem menampilkan tombol "Tambah Akun"
4. Admin meng-klik tombol "Tambah Akun"
5. Sistem menampilkan formulir pengisian data Siswa
6. Admin mengisi formulir data siswa:
    - Nama Lengkap
    - Tanggal Lahir
    - NIM/NISN
    - Email Aktif
    - Nama Orang Tua/Wali
    - Surat Persetujuan Orang Tua
    - Password
7. Sistem memvalidasi data
8. Sistem membuat akun
9. Sistem mengarahkan pengguna kembali ke halaman Manajemen Akun dan menampilakn notifikasi bahwa akun siswa berhasil dibuat

**Altenative Flow (Optional Path)**
- **A1** : Admin mendapatkan link yang mengarahkan ke formulir pendaftaran mandiri lalu menyebarkannya ke seluruh Siswa/Mahasiswa untuk mendaftar.

**Exception Flow (Error Handling)**
- **E1** : Email sudah terdaftar -> Sistem menampilkan pesan "Email ini sudah terdaftar untuk akun ```nama_siswa```"