import React from 'react'
import styled from 'styled-components'
import { theme } from '../theme'

const HeaderContainer = styled.header`
  background: linear-gradient(135deg, ${theme.colors.purple}, ${theme.colors.teal});
  padding: ${theme.spacing.xl} ${theme.spacing.lg};
  text-align: center;
  border-bottom: 3px solid ${theme.colors.mint};
  box-shadow: ${theme.shadows.lg};
`

const Title = styled.h1`
  font-size: 2.5rem;
  color: ${theme.colors.light};
  margin-bottom: ${theme.spacing.md};
  letter-spacing: 1px;
  
  @media (max-width: ${theme.breakpoints.mobile}) {
    font-size: 1.8rem;
  }
`

const Subtitle = styled.p`
  font-size: 1.3rem;
  color: ${theme.colors.light};
  margin-bottom: ${theme.spacing.lg};
  opacity: 0.95;
  
  @media (max-width: ${theme.breakpoints.mobile}) {
    font-size: 1rem;
  }
`

const AuthorInfo = styled.div`
  background: rgba(23, 34, 38, 0.8);
  padding: ${theme.spacing.lg};
  border-radius: ${theme.borderRadius.md};
  max-width: 800px;
  margin: 0 auto;
  border: 2px solid ${theme.colors.mint};
`

const AuthorName = styled.h2`
  color: ${theme.colors.mint};
  font-size: 1.5rem;
  margin-bottom: ${theme.spacing.sm};
`

const AuthorTitle = styled.p`
  color: ${theme.colors.light};
  font-size: 1.1rem;
  margin-bottom: ${theme.spacing.sm};
`

const AuthorEducation = styled.p`
  color: ${theme.colors.light};
  font-size: 0.95rem;
  opacity: 0.9;
`

const MetaGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${theme.spacing.md};
  margin-top: ${theme.spacing.lg};
  
  @media (max-width: ${theme.breakpoints.mobile}) {
    grid-template-columns: 1fr;
  }
`

const MetaCard = styled.div`
  background: linear-gradient(135deg, ${theme.colors.teal}, ${theme.colors.purple});
  padding: ${theme.spacing.md};
  border-radius: ${theme.borderRadius.md};
  border-left: 4px solid ${theme.colors.mint};
  text-align: center;
`

const MetaLabel = styled.div`
  color: ${theme.colors.light};
  font-size: 0.9rem;
  margin-bottom: ${theme.spacing.xs};
  opacity: 0.8;
`

const MetaValue = styled.div`
  color: ${theme.colors.light};
  font-size: 1.5rem;
  font-weight: bold;
`

const Header = () => {
  return (
    <HeaderContainer>
      <Title>Nobel Prize Winners Analysis Dashboard</Title>
      <Subtitle>Interactive Multi-Dimensional Data Science Analysis (1901-2025)</Subtitle>
      
      <AuthorInfo>
        <AuthorName>Cazandra Aporbo, MS</AuthorName>
        <AuthorTitle>Head of Data Science, FoXX Health</AuthorTitle>
        <AuthorEducation>
          MS in Data Science, University of Denver | BS in Integrative Biology, Oregon State University
        </AuthorEducation>
      </AuthorInfo>

      <MetaGrid>
        <MetaCard>
          <MetaLabel>Total Awards</MetaLabel>
          <MetaValue>995</MetaValue>
        </MetaCard>
        <MetaCard>
          <MetaLabel>Time Period</MetaLabel>
          <MetaValue>125 Years</MetaValue>
        </MetaCard>
        <MetaCard>
          <MetaLabel>Countries</MetaLabel>
          <MetaValue>76</MetaValue>
        </MetaCard>
        <MetaCard>
          <MetaLabel>Categories</MetaLabel>
          <MetaValue>6</MetaValue>
        </MetaCard>
      </MetaGrid>
    </HeaderContainer>
  )
}

export default Header
