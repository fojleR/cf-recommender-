export const metadata = {
  title: 'Codeforces Problem Recommender',
  description: 'Get personalized Codeforces problem recommendations based on your handle.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}